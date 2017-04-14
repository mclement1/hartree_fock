// C++ headers
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <math.h>

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
//#include <unsupported/Eigen/MatrixFunctions>

// Libint gaussian integrals library
#include <libint2.hpp>

using std::cout;
using std::cerr;
using std::endl;
using std::size_t;
using std::vector;
using std::tuple;


using libint2::BasisSet;
using libint2::Atom;
using libint2::Shell;
using libint2::Engine;
using libint2::Operator;


//*******************************************************************************
// General code
//*******************************************************************************


// Create a class, Tensor4
namespace mcc {

template <typename T>
class Tensor4 {
  public:
    Tensor4() = default;  // default constructor

    // size_t better than in bc size_t is 64 bit and unsigned
    Tensor4(size_t n0, size_t n1, size_t n2, size_t n3) : data_(n0*n1*n2*n3) {
    n_[0] = n0; // private data members get underscores at end of name
    n_[1] = n1;
    n_[2] = n2;
    n_[3] = n3;
    }
    ~Tensor4() = default; // compiler takes care of cleanup

    T operator()(size_t idx0, size_t idx1, size_t idx2, size_t idx3) const { // const means func doesn't modify
      return data_.at(((idx0*n_[1] + idx1)*n_[2] + idx2)*n_[3] + idx3);
    }

    // operator() means that operator is called with parentheses; could use square bracket, =, etc.    
    T& operator()(size_t idx0, size_t idx1, size_t idx2, size_t idx3) { 
      return data_.at(((idx0*n_[1] + idx1)*n_[2] + idx2)*n_[3] + idx3);
    }
    
  private:
    std::vector<T> data_; // dense storage for the data
    size_t n_[4]; // extents
  };
}

// Create a class, Tensor3
namespace mcc {

template <typename T>
class Tensor3 {
  public:
    Tensor3() = default;  // default constructor

    // size_t better than in bc size_t is 64 bit and unsigned
    Tensor3(size_t n0, size_t n1, size_t n2) : data_(n0*n1*n2) {
    n_[0] = n0; // private data members get underscores at end of name
    n_[1] = n1;
    n_[2] = n2;
    //n_[3] = n3;
    }
    ~Tensor3() = default; // compiler takes care of cleanup

    T operator()(size_t idx0, size_t idx1, size_t idx2) const { // const means func doesn't modify
      return data_.at((idx0*n_[1] + idx1)*n_[2] + idx2);
    }

    // operator() means that operator is called with parentheses; could use square bracket, =, etc.    
    T& operator()(size_t idx0, size_t idx1, size_t idx2) { 
      return data_.at((idx0*n_[1] + idx1)*n_[2] + idx2);
    }
    
  private:
    std::vector<T> data_; // dense storage for the data
    size_t n_[3]; // extents
  };
}




// Define a Matrix type 
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;


// Location of geometry file and basis set file
std::string coords = "/Users/mcclement/practice/hartree_fock/geom/ch4-ang.xyz";
std::string basis_set = "cc-pvdz";


//*******************************************************************************
// Hartree Fock 
//*******************************************************************************

// Read in molecular coordinates
std::vector<Atom> read_geom(std::string coords) {
  std::string molecule = coords;
  std::ifstream input_file(molecule);
  std::vector<Atom> atoms = libint2::read_dotxyz(input_file);

  return atoms;
}


// Determine total number of electrons and occupied orbitals
int count_elec(std::vector<Atom> atoms) {
  int nelec = 0;
  for (int i=0; i<atoms.size(); ++i) {
    nelec += atoms[i].atomic_number;
  } 
  int num_occ;
  if (nelec % 2 == 0)
    num_occ = nelec/2;
  else
    num_occ = (nelec + 1)/2;
  return num_occ;
}


//Form the initial electron density matrix, P

//Matrix make_p(BasisSet basis) {
Matrix make_p(int nao) {
  //int nao = basis.nbf();
   Matrix p(nao, nao);
    
    for (int i=0; i<nao; ++i) {
      for (int j=0; j<nao; ++j) {
        if (i!=j)
          p(i, j) = 0;
        else 
          if (i == 0 || i == 1) 
            p(i, i) = 1;
          else if (i == nao-1 || i == nao-2)
            p(i, i) = 0.5;
          else
            p(i, i) = (2.0/3.0);
      }
    }
  return p;
}

// Function to update the density matrix P
Matrix update_p(int nao, int nocc, Matrix C) {

  Matrix newP(nao, nao);

  for (int i=0; i<nao; ++i) {
    for (int j=0; j<nao; ++j) {
      double ij = 0.0;
      for (int k=0; k<nocc; ++k) {
        ij += C(i,k)*C(j,k);
      }
      newP(i,j) = ij;
    }
  }
  return newP;
}




// Compute the nuclear repulsion energy
double comp_E_nuc (std::vector<Atom> atoms) {

  double nuc_rep = 0.0;
  for (int j=1; j<atoms.size(); ++j) {
    for (int i=0; i<j; ++i) {
      double xij = atoms[i].x - atoms[j].x;
      double yij = atoms[i].y - atoms[j].y;
      double zij = atoms[i].z - atoms[j].z;
      double rij = xij*xij + yij*yij + zij*zij;     
      double r = std::sqrt(rij);
      nuc_rep += (atoms[i].atomic_number*atoms[j].atomic_number)/r;
    }
  }
  return nuc_rep;
}


//Compute one-electron integrals (nuclear attraction,
// kinetic energy, and overlap) and store in a Matrix 

Matrix one_elec_compute(BasisSet basis, int nao, Operator op, std::vector<Atom> atoms) {

  // Matrix dimensions
  //int nao = basis.nbf();

  // Define uninitialized  matrix of appropriate dimensions
  Matrix integral_mat(nao,nao);

  // Create one electron integral engine
  Engine one_elec_engine(op,
            basis.max_nprim(),
            basis.max_l()
            );

  if (op == Operator::nuclear) {
    one_elec_engine.set_params(make_point_charges(atoms));
  }

  // Map shell index to basis function index
  auto shell2bf = basis.shell2bf();

  // Point to each computed shell set
  const auto& buf_vec = one_elec_engine.results();

  // Loop over unique pairs of functions
  for (auto s1=0; s1!=basis.size(); ++s1) {

    auto bf1 = shell2bf[s1]; // (absolute) index of first basis func in shell s1
    auto n1 = basis[s1].size(); // number of basis func in shell s1

    for(auto s2=0; s2<=s1; ++s2) {
    
      auto bf2 = shell2bf[s2]; // index of first basis func in shell s2
      auto n2 = basis[s2].size(); // number of basis func in shell s2
   
      // Compute integral
      one_elec_engine.compute(basis[s1], basis[s2]);

      // Store integral value in uninitialized Matrix
      Eigen::Map<const Matrix> buf_mat(buf_vec[0], n1, n2); 
      integral_mat.block(bf1, bf2, n1, n2 ) = buf_mat;
    
      if(s1!=s2)
        integral_mat.block(bf2, bf1, n2, n1) = buf_mat.transpose();
    }
  }
  return integral_mat;
}


// Diagonalize the overlap matrix S to form 
// the transformation matrix X

Matrix diagonalize_s(Matrix S, int nao) {
  Eigen::SelfAdjointEigenSolver<Matrix> es;
  es.compute(S);
  Matrix s = es.eigenvalues().asDiagonal();
  //cout << "Matrix s: \n\n" << s << "\n" << endl;
  Matrix spow(nao, nao);
  for (int i=0; i<nao; ++i) {
    for (int j=0; j<nao; ++j ){
      if (i == j)
        spow(i, j) = 1/sqrt(s(i, j));
      else
        spow(i, j) = s(i, j);
    }
  }      

  Matrix U = es.eigenvectors();
  Matrix X = U*spow*U.adjoint();
  return X;
}

// Compute the two body eris and store in a four index tensor
mcc::Tensor4<double> comp_ao_eri(BasisSet basis) {

  size_t nao = basis.nbf();
  mcc::Tensor4<double> ao_eris(nao, nao, nao, nao);

  // Create two-electron integral engine
  Engine eri_engine(Operator::coulomb, basis.max_nprim(), basis.max_l());

  // Map shell index to basis function index
  auto shell2bf = basis.shell2bf();

  // Point to each computed shell set
  const auto& buf_vec = eri_engine.results();

  for (int s1=0; s1<basis.size(); ++s1) {
    int bf1 = shell2bf[s1];
    int n1 = basis[s1].size();

    for (int s2=0; s2<basis.size(); ++s2) {
      int bf2 = shell2bf[s2];
      int n2 = basis[s2].size();

      for (int s3=0; s3<basis.size(); ++s3) {
        int bf3 = shell2bf[s3];
        int n3 = basis[s3].size();

        for (int s4=0; s4<basis.size(); ++s4) {
          int bf4 = shell2bf[s4];
          int n4 = basis[s4].size();


          // Compute the eris
          eri_engine.compute(basis[s1], basis[s2], basis[s3], basis[s4]);            

          // Location of computed shell set of coulomb integrals
          const auto* buf_1234 = buf_vec[0];

          for (int f1=0; f1<n1; ++f1) {
            for (int f2=0; f2<n2; ++f2) {
              for (int f3=0; f3<n3; ++f3) {
                for (int f4=0; f4<n4; ++f4) {
                  ao_eris(bf1+f1, bf2+f2, bf3+f3, bf4+f4) = buf_1234[f1*n2*n3*n4+f2*n3*n4+f3*n4+f4];
                }
              }
            }
          }
        }
      }
    }
  }
  return ao_eris;

}


// Form the coulomb matrix
Matrix form_coulomb(int nao, mcc::Tensor4<double> ao_eris, Matrix density) {

  // Declare coulomb matrix
  Matrix J(nao, nao);

  for (int f1=0; f1<nao; ++f1) {

    for (int f2=0; f2<nao; ++f2) {
      double sum = 0.0;

      for (int f3=0; f3<nao; ++f3) {

        for (int f4=0; f4<nao; ++f4) {
          sum += ao_eris(f1, f2, f3, f4)*density(f3, f4);
        }
      }
      J(f1, f2) = sum;
    }  
  }
  return J;
}


// Form the exchange matrix
Matrix form_exchange(int nao, mcc::Tensor4<double> ao_eris, Matrix density) {

  // Declare exchange matrix
  Matrix K(nao, nao);

  for (int f1=0; f1<nao; ++f1) {

    for (int f2=0; f2<nao; ++f2) {
      double sum = 0.0;

      for (int f3=0; f3<nao; ++f3) {

        for (int f4=0; f4<nao; ++f4) {
          sum += ao_eris(f1, f4, f3, f2)*density(f4, f3);
        }
      }
      K(f1, f2) = sum;
    }  
  }
  return K;
}

// Compute the electronic energy
double comp_E_elec(int nao, Matrix H, Matrix F, Matrix P) {

  double E_elec = 0.0;

  for (int i=0; i<nao; ++i) {
    for (int j=0; j<nao; ++j) {
      E_elec += (H(i,j) + F(i,j))*P(i,j);
    }
  }
  return E_elec;
} 




// Loop to carry out Hartree Fock process
tuple<int, int, Matrix, Matrix, mcc::Tensor4<double>> hf_proc(std::string coords, std::string basis_set) {

  tuple<int, int, Matrix, Matrix, mcc::Tensor4<double>> hf_vals;

  //Read in moleculare geom
  std::vector<Atom> atoms = read_geom(coords);
  
  // Create the basis set object
  BasisSet basis(basis_set, atoms);

  // Determine number of AOs (i.e., # of basis functions)
  int nao = basis.nbf();

  // Determine number of occupied AOs
  int nocc = count_elec(atoms);

  // Determine the number of virtual AOs
  int nvirt = nao - nocc;

  // Calculate the nuclear repulsion energy
  double E_nuc = comp_E_nuc(atoms);

  // Compute all 2-electron eris
  mcc::Tensor4<double> ao_eris = comp_ao_eri(basis);

  // Form the overlap (S) matrix
  Matrix S = one_elec_compute(basis, nao, Operator::overlap, atoms);
  
  // Diagonalize the S matrix to form the transformation matrix X
  Matrix X = diagonalize_s(S, nao);

  // Form the kinetic energy (T) matrix
  Matrix T = one_elec_compute(basis, nao, Operator::kinetic, atoms);

  // Form the kinetic energy (V) matrix
  Matrix V = one_elec_compute(basis, nao, Operator::nuclear, atoms);

  // Form the core Hamiltonian (H) matrix
  Matrix H = T + V;

  // Form the initial electron density (P) matrix
  Matrix P = make_p(nao);

  // Form the initial coulomb (J) matrix
  Matrix J = form_coulomb(nao, ao_eris, P);
  //cout << "\nThe initial J matrix:\n" << J << endl;

  // Form the initial exchange (K) matrix
  Matrix K = form_exchange(nao, ao_eris, P); 
  //cout << "\nThe initial K matrix:\n" << K << endl;

  // Form the intial Fock (F) matrix
  Matrix F = H + 2*J - K;
  //cout << "\nThe initial Fock (F) matrix:\n" << F << endl;  

  // Declare the initial C matrix
  Matrix C(nao, nao);

  // Declare a matrix to hold the epsilon values
  Matrix epsilon(nao, nao);


  // Main iterative loop

  // Initialize the frobenius value
  double frob = 1.0;

  // Initialize a count of the iterations
  int iter = 0;

  // Set convergence value
  double conv = 1.0e-7;

  // Set number of max iterations
  int max_iter = 100;

  while (frob > conv && iter < max_iter) {

    // Calculate the transformed Fock (F') matrix
    Matrix Fprime = X*F*X;

    // Diagonalize F' to obtain C' and epsilon
    Eigen::SelfAdjointEigenSolver<Matrix> es;
    es.compute(Fprime);
    Matrix Cprime = es.eigenvectors();
    epsilon = es.eigenvalues().asDiagonal();

    // Calculate C from C'
    //Matrix C = X*Cprime;
    C = X*Cprime;

    // Form the new density matrix
    Matrix newP = update_p(nao, nocc, C);

    // Compute diff between new and old P matrices
    // and compute frobenius norm
    Matrix diff = newP - P;
    frob = diff.norm();
    P = newP;
    J = form_coulomb(nao, ao_eris, P);
    K = form_exchange(nao, ao_eris, P);
    F = H + 2*J - K;
    iter += 1;
  }

  // Compute the electronic energy
  double E_elec = comp_E_elec(nao, H, F, P);

  // Compute the Hartree Fock energy
  double EHF = E_nuc + E_elec;

  // Print important quantities
  cout << "\nThe electronic energy is " << E_elec << endl;
  cout << "The nuclear repulsion enery is " << E_nuc << endl;
  cout << "The Hartree Fock energy is " << EHF << endl;
  cout << "Number of iterations: " << iter << endl;
  //cout << "The final C matrix:\n" << C << endl;
  //cout << "The matrix of epsilon values:\n" << epsilon << endl;
  // Return the following quantities for use in MP2:
  // nao, nocc, C, epsilon, ao_ints
  hf_vals = std::tie(nao, nocc, C, epsilon, ao_eris);

  return hf_vals;
}


//*******************************************************************************
// MP2 
//*******************************************************************************

// Original contraction code
//// Contract over fourth index
//mcc::Tensor3<double> contract_over_fourth(int nao, int i, mcc::Tensor4<double> ao_eris,
//                                          Matrix C) {
//
//  mcc::Tensor3<double> three_center(nao, nao, nao);
//
//  for (int f1=0; f1<nao; ++f1) {
//    for (int f2=0; f2<nao; ++f2) {
//      for (int f3=0; f3<nao; ++f3) { 
//        double sum = 0.0;
//
//        for (int f4=0; f4<nao; ++f4) {
//          sum += ao_eris(f1, f2, f3, f4)*C(f4, i);
//        }
//        three_center(f1, f2, f3) = sum;
//      }
//    }
//  }
//  return three_center;
//}
//
//
//// Contract over third index
//Matrix contract_over_third(int nao, int i, mcc::Tensor3<double> three_center, Matrix C) {
//
//  Matrix two_center(nao, nao);
//
//  for (int f1=0; f1<nao; ++f1) {
//    for (int f2=0; f2<nao; ++f2) {
//      double sum = 0.0;
//
//      for (int f3=0; f3<nao; ++f3) {
//        sum += three_center(f1, f2, f3)*C(f3, i);
//      }
//      two_center(f1, f2) = sum;
//    }
//  }
//  return two_center;
//}
//
//
//// Contract over second index
//vector<double> contract_over_second(int nao, int i, Matrix two_center, Matrix C) {
//
//  vector<double> one_center(nao);
//
//  for (int f1=0; f1<nao; ++f1) {
//    double sum = 0.0;
//
//    for (int f2=0; f2<nao; ++f2) {
//      sum += two_center(f1, f2)*C(f2, i);
//    }
//    one_center[f1] = sum;
//  }
//  return one_center;
//}
//
//// Contract over first index
//double contract_over_first(int nao, int i, vector<double> one_center, Matrix C) {
//
//  double mo_int = 0.0;
//  for (int f1=0; f1<nao; ++f1) {
//    mo_int += one_center[f1]*C(f1, i);
//  }
//  return mo_int;
//}


// Modified contraction code
// Contract over fourth index. Result in 4_D tensor containing
// elements of the form (ao ao | ao mo)
mcc::Tensor4<double> contract_over_fourth(int nao, mcc::Tensor4<double> ao_eris,
                                          Matrix C) {

  mcc::Tensor4<double> ao_ao_ao_mo(nao, nao, nao, nao);

  for (int f1=0; f1<nao; ++f1) {
    for (int f2=0; f2<nao; ++f2) {
      for (int f3=0; f3<nao; ++f3) { 

        for (int i=0; i<nao; ++i) { // MO index
          double sum = 0.0;
          for (int f4=0; f4<nao; ++f4) {
            sum += ao_eris(f1, f2, f3, f4)*C(f4, i);
          }
          ao_ao_ao_mo(f1, f2, f3, i) = sum;
        }
      }
    }
  }
  return ao_ao_ao_mo;
}


// Contract over third index. Result is a 4-D tensor containing
// elements of the form (ao ao | mo mo)
mcc::Tensor4<double> contract_over_third(int nao, mcc::Tensor4<double> ao_ao_ao_mo, Matrix C) {

  mcc::Tensor4<double> ao_ao_mo_mo(nao, nao, nao, nao);

  for (int f4=0; f4<nao; ++f4) {

    for (int f1=0; f1<nao; ++f1) {
      for (int f2=0; f2<nao; ++f2) {

        for (int i=0; i<nao; ++i) {
          double sum = 0.0;

          for (int f3=0; f3<nao; ++f3) {
            sum += ao_ao_ao_mo(f1, f2, f3, f4)*C(f3, i);
          }
          ao_ao_mo_mo(f1, f2, i, f4) = sum;
        }
      }
    }
  }
  return ao_ao_mo_mo;
}


// Contract over second index
// Result if a 4-D tensor containing elements of the form (ao mo | mo mo)
mcc::Tensor4<double> contract_over_second(int nao, mcc::Tensor4<double> ao_ao_mo_mo, Matrix C) {

  mcc::Tensor4<double> ao_mo_mo_mo(nao, nao, nao, nao);

  for (int f4=0; f4<nao; ++f4) {
    for (int f3=0; f3<nao; ++f3) {
      for (int f1=0; f1<nao; ++f1) {
        for (int i=0; i<nao; ++i) {
          double sum = 0.0;
          for (int f2=0; f2<nao; ++f2) {
            sum += ao_ao_mo_mo(f1, f2, f3, f4)*C(f2, i);
          }
          ao_mo_mo_mo(f1, i, f3, f4) = sum;
        }
      }
    }
  }
  return ao_mo_mo_mo;
}

// Contract over first index
// Result is a 4_D tensor containing elements of the form (mo mo | mo mo)
mcc::Tensor4<double> contract_over_first(int nao, mcc::Tensor4<double> ao_mo_mo_mo, Matrix C) {

  mcc::Tensor4<double> mo_mo_mo_mo(nao, nao, nao, nao);

  for (int f4=0; f4<nao; ++f4) {
    for (int f3=0; f3<nao; ++f3) {
      for (int f2=0; f2<nao; ++f2) {
        for (int i=0; i<nao; ++i) {
          double sum = 0.0;
          for (int f1=0; f1<nao; ++f1) {
            sum += ao_mo_mo_mo(f1, f2, f3, f4)*C(f1,i);
          }
          mo_mo_mo_mo(i, f2, f3, f4) = sum;
        }
      }
    }
  }
  return mo_mo_mo_mo;
}



// Loop to make mo_mo_mo_mo tensor
mcc::Tensor4<double> comp_mo_eris(int nao, mcc::Tensor4<double> ao_eris, Matrix C) {

  mcc::Tensor4<double> ao_ao_ao_mo = contract_over_fourth(nao, ao_eris, C);
  mcc::Tensor4<double> ao_ao_mo_mo = contract_over_third(nao, ao_ao_ao_mo, C);
  mcc::Tensor4<double> ao_mo_mo_mo = contract_over_second(nao, ao_ao_mo_mo, C);
  mcc::Tensor4<double> mo_mo_mo_mo = contract_over_first(nao, ao_mo_mo_mo, C);

  return mo_mo_mo_mo;
}

// For combination i, j, a, b form MP2 term
double mp2_term(int i, int j, int a, int b, mcc::Tensor4<double> mo_mo_mo_mo, Matrix eps) {

  double first_int = mo_mo_mo_mo(i, a, j, b);
  double second_int = mo_mo_mo_mo(i, b, j, a);
  double num = first_int*(2*first_int - second_int);
  double denom = eps(i,i) + eps(j,j) - eps(a,a) - eps(b,b);
  double term = num/denom;
  return term;
}


// Loop over every combination of i, j, a, b
double comp_E_mp2 (int nocc, int nao, mcc::Tensor4<double> mo_mo_mo_mo, Matrix eps) {

  double sum = 0.0;

  for (int i=0; i<nocc; ++i) {
    for (int j=0; j<nocc; ++j) {
      for (int a=nocc; a<nao; ++a) {
        for (int b=nocc; b<nao; ++b) {
          double term = mp2_term(i, j, a, b, mo_mo_mo_mo, eps);
          sum += term;
        }
      }
    }
  }
  return sum;
}


// Loop for MP2 process
int mp2_proc(std::tuple<int, int, Matrix, Matrix, mcc::Tensor4<double>> hf_vals) {

   int nao = std::get<0> (hf_vals);
   int nocc = std::get<1> (hf_vals);
   Matrix C = std::get<2> (hf_vals);
   Matrix eps = std::get<3> (hf_vals);
   mcc::Tensor4<double> ao_eris = std::get<4> (hf_vals);
   mcc::Tensor4<double> mo_mo_mo_mo = comp_mo_eris(nao, ao_eris, C);

  double E_mp2 = comp_E_mp2(nocc, nao, mo_mo_mo_mo, eps);
  cout << "The MP2 correction to the energy is " << E_mp2 << endl;

  return 0;
} 

//
//// Loop for contracting over fourth index 
//// Result is 4-D tensor containing integrals of form (ao ao | ao mo)
//mcc::Tensor4<duoble> fourth_contract_loop(int nao, mcc::Tensor4<double> ao_eris, Matrix C) {
//
//  //vector<mcc::Tensor3<double>> fourth_contract(nao);
//  mcc::Tensor4<double> fourth_contract(nao, nao, nao, nao);
//  for (int i=0; i<nao; ++i) {
//    mcc::Tensor3<double> three_center = contract_over_fourth(nao, i, ao_eris, C);
//    fourth_contract[i] = three_center;
//  }  
//
//  return fourth_contract;
//}
//
//
//// Loop for contracting over fourth index
//Matrix third_constract_loop(int nao, vector<mcc::Tensor3<double>> fourth_contract, Matrix C) {
//
//  for (int i=0; i<nao; ++i) {
//    mcc::Tenrsor




//
//// contract over fourth index for every b
//vector<mcc::Tensor3<double>> contract_fourth_loop() {
//
//  vector<mcc::Tensor3<double>> fourth_contract;
//  for int(b=nocc; b<nao; ++b) {
//    mcc::Tensor3<double> three_center = contract_over_fourth(nao, b, ao_eris, C);
//    fourth_contract[b-nocc] = three_center;
//  }
//  return fourth_contract;
//}
//
//
//// Contract over third index for every j
//vector<Matrix> contract_third_loop() {
//
//  vector<Matrix> third_contract;
//  for (int b=nocc; b<nao; ++b) {
//    double bp = b-nocc;
//    
//    for (int j=0; j<nocc; ++j) {
      


//
//// For combination of i, j, a, b form first int
//double comp_first_int(int i, int a, int j, int b, int nao, mcc::Tensor4<double> ao_eris, Matrix C) {
//
//  mcc::Tensor3<double> three_center = contract_over_fourth(nao, b, ao_eris, C);
//  Matrix two_center = contract_over_third(nao, j, three_center, C);
//  vector<double> one_center = contract_over_second(nao, a, two_center, C);
//  double mo_int = contract_over_first(nao, i, one_center, C);
//
//  return mo_int;
//}
//
//// For combination of i, j, a, b form second int
//double comp_second_int(int i, int b, int j, int a, int nao, mcc::Tensor4<double> ao_eris, Matrix C) {
//
//  mcc::Tensor3<double> three_center = contract_over_fourth(nao, a, ao_eris, C);
//  Matrix two_center = contract_over_third(nao, j, three_center, C);
//  vector<double> one_center = contract_over_second(nao, b, two_center, C);
//  double mo_int = contract_over_first(nao, i, one_center, C);
//
//  return mo_int;
//}
//  
//// For combination of i, j, a, b form MP2 term
//double comp_mp2_term(int i, int j, int a, int b, int nao, mcc::Tensor4<double> ao_eris, Matrix C, Matrix eps) {
//
//  double first_int = comp_first_int(i, a, j, b, nao, ao_eris, C);
//  double second_int = comp_second_int(i, b, j, a, nao, ao_eris, C);
//  double num = first_int*(2*first_int - second_int);
//  double denom = eps(i,i) + eps(j,j) - eps(a,a) - eps(b,b);
//  double term = num/denom;
//  //cout << "first int = " << first_int << endl;
//  //cout << "second int = " << second_int << endl;
//  return term;
//} 
//
//
//// Compute the MP2 correction to the energy
////double comp_E_mp2(int nao, int nocc, mcc::Tensor4<double> ao_eris, Matrix C, Matrix eps) {
//double comp_E_mp2(tuple<int, int, Matrix, Matrix, mcc::Tensor4<double>> hf_vals) {
//
//  int nao = std::get<0> (hf_vals);
//  int nocc = std::get<1> (hf_vals);
//  Matrix C = std::get<2> (hf_vals);
//  Matrix eps = std::get<3> (hf_vals);
//  mcc::Tensor4<double> ao_eris = std::get<4> (hf_vals);
// 
//  cout << "nao = " << nao << endl;
//  cout << "nocc = " << nocc << endl;
//  //cout << "C:\n" << C << endl;
//  //cout << "eps:\n" << eps << endl;
//
//  double E_mp2 = 0.0;
//  for (int b=nocc; b<nao; ++b) {
//    for (int a=nocc; a<nao; ++a) {
//      for (int j=0; j<nocc; ++j) {
//        for (int i=0; i<nocc; ++i) {
//          double term = comp_mp2_term(i, j, a, b, nao, ao_eris, C, eps);
//          E_mp2 += term;
//          //cout << "E " << E_mp2 << endl;
//        }
//      }
//    }
//  }
//  //double E_mp2 = 0.0;
//  //for (int i=0; i<nocc; ++i) {
//  //  for (int j=0; j<nocc; ++j) {
//  //    for (int a=nocc; a<nao; ++a) {
//  //      for (int b=nocc; b<nao; ++b) {
//  //        double term = comp_mp2_term(i, j, a, b, nao, ao_eris, C, eps);
//  //        E_mp2 += term;
//  //        //cout << "E " << E_mp2 << endl;
//  //      }
//  //    }
//  //  }
//  //}
//  cout << "The MP2 correction to the energy is " << E_mp2 << endl;
//  return E_mp2;
//}         



//*******************************************************************************
// Pair Natural Orbitals PNOs 
//*******************************************************************************

//
//// Form the K ij mu nu matrix by contracting over occ AOs
//Matrix form_K_munu(int nao, int i, int j, mcc::Tensor4<double> ao_eris, Matrix C) {
//
//  // Declare K^mu,nu matrix
//  Matrix Kmunu(nao, nao);
//
//  for (int f1=0; f1<nao; ++f1) {
//
//    for (int f2=0; f2<nao; ++f2) {
//      double sum = 0.0;
//
//      for (int f3=0; f3<nao; ++f3) {
//
//        for (int f4=0; f4<nao; ++f4) {
//          sum += ao_eris(f1, f4, f3, f2)*C(f3, a)*C(f4, b);
//        }
//      }
//      Kmunu(f1, f2) = sum;
//    }  
//  }
//  return Kmunu;
//}
//
//// Form the K^ij _ab matrix
//Matrix form_K_ij(int nao, int i, int j, Matrix Kmunu, Matrix C) {
//
//  // Declare K^ij matrix
//  Matrix Kij(nao, nao);
//
//  for (int f1=0; f1<nao; ++f1) {
//    for (int f2=0; f2<nao; ++f2) {
//      Kij(f1, f2) = Kmunu(f1, f2)*C(f1, i)*C(f2, j);
//    }
//  }
//  return Kij;
//}
//
//
//
//// Form the T^ij matrix
//Matrix form_Tij() {
//
//  Matrix Kmunu = form_K_munu(nao, a, b, )
//
//
//}
//
//
//// Form the T^ji matix
//Matrix form_Tji() {
//
//
//
//}


int main() {

  libint2::initialize();

  //cout << "Hello, World!" << endl;
  tuple<int, int, Matrix, Matrix, mcc::Tensor4<double>>  hf_vals = hf_proc(coords, basis_set);
  int E_mp2 = mp2_proc(hf_vals);

  libint2::finalize();

  return 0;
} 
