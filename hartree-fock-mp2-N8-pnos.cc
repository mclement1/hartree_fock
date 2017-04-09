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
#include <unsupported/Eigen/MatrixFunctions>

// Libint gaussian integrals library
#include <libint2.hpp>

using std::cout;
using std::cerr;
using std::endl;
using std::vector;


using libint2::BasisSet;
using libint2::Atom;
using libint2::Shell;
using libint2::Engine;
using libint2::Operator;

// Location of geometry file and basis set file
std::string COORDS = "/Users/mcclement/practice/hartree_fock/geom/h2o.xyz";
//std::string NAME = "h2o_cc-pvdz.out";
std::string BASIS_SET = "sto-3g";

// Conversion factor: 1 bohr = 0.52917721067e-10 angstrom
double conversion = 0.52917721067e-10;




// Define a Matrix type 
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;


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


// Create basis set object
BasisSet create_bs(std::string basis_set, std::vector<Atom> atoms) {
    
  BasisSet basis(basis_set, atoms);
  return basis;
}

// Determine total number of basis functions
int tot_fun(BasisSet basis) {
  int total_sum = 0;
  for (int s=0; s<basis.size(); ++s) {
    int part_sum = basis[s].size();
    total_sum += part_sum;
  }
  return total_sum;
}


// Determine total number of basis functions
int sum_func(BasisSet basis) {
  int num_func=0;
  for (int s=0; s!=basis.size(); ++s) {
    Shell shell = basis[s];
    int am = shell.contr[0].l;
    int num = ((am + 1)*(am + 2))/2;
    num_func += num;
  }
  return num_func;
}


//Form the initial electron density matrix, P

Matrix make_p(int num_func) {
    int n = num_func;
   Matrix p(n,n);
    
    for (int i=0; i<n; ++i) {
      for (int j=0; j<n; ++j) {
        if (i!=j)
          p(i, j) = 0;
        else 
          if (i == 0 || i == 1) 
            p(i, i) = 1;
          else if (i == n-1 || i == n-2)
            p(i, i) = 0.5;
          else
            p(i, i) = (2.0/3.0);
       }
    }
  return p;
}


// computes Superposition-Of-Atomic-Densities guess for the molecular density matrix
// in minimal basis; occupies subshells by smearing electrons evenly over the orbitals
Matrix compute_soad(const std::vector<Atom>& atoms) {

  // compute number of atomic orbitals
  size_t nao = 0;
  for(const auto& atom: atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) // H, He
      nao += 1;
    else if (Z <= 10) // Li - Ne
      nao += 5;
    else
      throw "SOAD with Z > 10 is not yet supported";
  }

  // compute the minimal basis density
  Matrix D = Matrix::Zero(nao, nao);
  size_t ao_offset = 0; // first AO of this atom
  for(const auto& atom: atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) { // H, He
      D(ao_offset, ao_offset) = Z; // all electrons go to the 1s
      ao_offset += 1;
    }
    else if (Z <= 10) {
      D(ao_offset, ao_offset) = 2; // 2 electrons go to the 1s
      D(ao_offset+1, ao_offset+1) = (Z == 3) ? 1 : 2; // Li? only 1 electron in 2s, else 2 electrons
      // smear the remaining electrons in 2p orbitals
      const double num_electrons_per_2p = (Z > 4) ? (double)(Z - 4)/3 : 0;
      for(auto xyz=0; xyz!=3; ++xyz)
        D(ao_offset+2+xyz, ao_offset+2+xyz) = num_electrons_per_2p;
      ao_offset += 5;
    }
  }

  return D * 0.5; // we use densities normalized to # of electrons/2
}



// Compute the nuclear repulsion energy
double nuc_rep (std::vector<Atom> atoms) {

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

Matrix one_elec_compute(BasisSet basis, int num_func, Operator op, std::vector<Atom> atoms) {

  // Matrix dimensions
  int n = num_func;

  // Define uninitialized  matrix of appropriate dimensions
  Matrix integral_mat(n,n);

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

Matrix diagonalize_s(Matrix S, int num_func) {
  Eigen::SelfAdjointEigenSolver<Matrix> es;
  es.compute(S);
  Matrix s = es.eigenvalues().asDiagonal();
  //cout << "Matrix s: \n\n" << s << "\n" << endl;
  Matrix spow(num_func, num_func);
  for (int i=0; i<num_func; ++i) {
    for (int j=0; j<num_func; ++j ){
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


// Compute the Coulomb eris and form the Coulomb (J) matrix

Matrix j_eri_compute(BasisSet basis, Matrix density_mat, int num_func) {

  // Set dimensions of J matrix
  int n = num_func;

  // Declare coulomb (J) matrix
  Matrix coulomb(n,n); 
  
  // Create two electron integral engine
  Engine eri_engine(Operator::coulomb,
            basis.max_nprim(),
            basis.max_l()
            );                  

  // Map shell index to basis function index
  auto shell2bf = basis.shell2bf();

  // Point to each computed shell set
  const auto& buf_vec = eri_engine.results();

  // For center 1, loop over shells
  for (auto s1=0; s1<basis.size(); ++s1) { 
    auto bf1 = shell2bf[s1]; // index of first bf in shell s1
    auto n1 = basis[s1].size(); // # of func in shell s1

    // Loop over every bf in shell
    for (auto f1=0; f1<n1; ++f1) {

      // For a given bf in a given shell on center 1,
      // loop over shells on center 2
      for (auto s2=0; s2<basis.size(); ++s2) {  
        auto bf2 = shell2bf[s2]; // index of first bf in shell s2
        auto n2 = basis[s2].size(); // # of func in shell s2

        // Loop over every function in shell
        for (auto f2=0; f2<n2; ++f2) {

          // Initialize sum for J corresponding to 
          // the bf on center 1 and the bf on center 2
          double J12 = 0.0;

          // For a given bf on center 1 and a given bf on
          // center 2, loop over shells on center 3
          for (auto s3=0; s3<basis.size(); ++s3) { 
            auto bf3 = shell2bf[s3]; // index of first bf in shell s3
            auto n3 = basis[s3].size(); // # of func in shell s3
           
            // For a given bf on center 1, a given bf on center 2
            // and a given shell on center 3, loop of shells on 
            // center 4
            for (auto s4=0; s4<basis.size(); ++s4) { 
                auto bf4 = shell2bf[s4]; // index of first bf in shell s4
                auto n4 = basis[s4].size(); // # of func in shell s4

                // Compute eri for coulomb contribution {s1, s2, s3, s4}
                eri_engine.compute(basis[s1], basis[s2], basis[s3], basis[s4]);                  
                //cout << bf1 << "," << bf2 << "," << bf3 << "," << bf4 << endl;
                
                // Location of computed (shell set) of coulomb integrals
                const auto* buf_1234 = buf_vec[0];
               
                // Sum together the eris corresponding to the particular
                // bf on center 1 and the particular bf on center 2
                auto d1 = n2*n3*n4; // # of bfs per func in s1
                auto d2 = n3*n4; // # of bfs per func in s2
                auto d3 = n4; // # of bfs per func in s3                   
                
                for (auto f3=0; f3<n3; ++f3) {
                  for (auto f4=0; f4<n4; ++f4) {
                    J12 = J12 + buf_1234[f1*d1+f2*d2+f3*d3+f4]*density_mat(bf3+f3, bf4+f4);
                  }
                }
              }
            }
            coulomb(bf1+f1, bf2+f2) = J12;
          }
        } 
      }
    }           
return coulomb;
}


// Compute the exchange eris and form the exchange (K) matrix

Matrix k_eri_compute(BasisSet basis, Matrix density_mat, int num_func) {

  // Set dimensions of K matrix
  int n = num_func;

  // Declare exchange (K) matrix
  Matrix exchange(n,n); 
  
  // Create two electron integral engine
  Engine eri_engine(Operator::coulomb,
            basis.max_nprim(),
            basis.max_l()
            );                  

  // Map shell index to basis function index
  auto shell2bf = basis.shell2bf();

  // Point to each computed shell set
  const auto& buf_vec = eri_engine.results();

  // For center 1, loop over shells
  for (auto s1=0; s1<basis.size(); ++s1) { 
    auto bf1 = shell2bf[s1]; // index of first bf in shell s1
    auto n1 = basis[s1].size(); // # of func in shell s1

    // Loop over every bf in shell
    for (auto f1=0; f1<n1; ++f1) {

      // For a given bf in a given shell on center 1,
      // loop over shells on center 2
      for (auto s2=0; s2<basis.size(); ++s2) {  
        auto bf2 = shell2bf[s2]; // index of first bf in shell s2
        auto n2 = basis[s2].size(); // # of func in shell s2

        // Loop over every function in shell
        for (auto f2=0; f2<n2; ++f2) {

          // Initialize sum for K corresponding to 
          // the bf on center 1 and the bf on center 2
          double K12 = 0.0;

          // For a given bf on center 1 and a given bf on
          // center 2, loop over shells on center 3
          for (auto s3=0; s3<basis.size(); ++s3) { 
            auto bf3 = shell2bf[s3]; // index of first bf in shell s3
            auto n3 = basis[s3].size(); // # of func in shell s3
           
            // For a given bf on center 1, a given bf on center 2
            // and a given shell on center 3, loop of shells on 
            // center 4
            for (auto s4=0; s4<basis.size(); ++s4) { 
                auto bf4 = shell2bf[s4]; // index of first bf in shell s4
                auto n4 = basis[s4].size(); // # of func in shell s4

                // Compute eri for exchange contribution {s1, s4, s3, s2}
                eri_engine.compute(basis[s1], basis[s4], basis[s3], basis[s2]);                  
                
                // Location of computed (shell set) of exchange integrals
                const auto* buf_1432 = buf_vec[0];
        
                // Sum together the eris corresponding to the particular
                // bf on center 1 and the particular bf on center 2
                auto d1 = n4*n3*n2;
                auto d2 = n3*n2;
                auto d3 = n2;
                
                for (auto f3=0; f3<n3; ++f3) {
                  for (auto f4=0; f4<n4; ++f4) {
                    K12 = K12 + buf_1432[f1*d1+f4*d2+f3*d3+f2]*density_mat(bf3+f3, bf4+f4);
                  }
                }
              }
            }
           exchange(bf1+f1, bf2+f2) = K12;
          }
        } 
      }
    }           
return exchange;
}



//*****************************************************************************
// Begin MP2 here
//*****************************************************************************



// Compute first integrals for MP2
Matrix mp2_compute_first(int j, int b, int num_func, BasisSet basis, Matrix C) {

  // Set dimensions of integral matrix
  int n = num_func;

  // Declare integral matrix
  Matrix first_integral(n,n); 
  
  // Create two electron integral engine
  Engine eri_engine(Operator::coulomb,
            basis.max_nprim(),
            basis.max_l()
            );                  

  // Map shell index to basis function index
  auto shell2bf = basis.shell2bf();

  // Point to each computed shell set
  const auto& buf_vec = eri_engine.results();

  // For center 1, loop over shells
  for (auto s1=0; s1<basis.size(); ++s1) { 
    auto bf1 = shell2bf[s1]; // index of first bf in shell s1
    auto n1 = basis[s1].size(); // # of func in shell s1

    // Loop over every bf in shell
    for (auto f1=0; f1<n1; ++f1) {

      // For a given bf in a given shell on center 1,
      // loop over shells on center 2
      for (auto s2=0; s2<basis.size(); ++s2) {  
        auto bf2 = shell2bf[s2]; // index of first bf in shell s2
        auto n2 = basis[s2].size(); // # of func in shell s2

        // Loop over every function in shell
        for (auto f2=0; f2<n2; ++f2) {

          // Initialize sum corresponding to 
          // the bf on center 1 and the bf on center 2
          double sum12 = 0.0;

          // For a given bf on center 1 and a given bf on
          // center 2, loop over shells on center 3
          for (auto s3=0; s3<basis.size(); ++s3) { 
            auto bf3 = shell2bf[s3]; // index of first bf in shell s3
            auto n3 = basis[s3].size(); // # of func in shell s3
           
            // For a given bf on center 1, a given bf on center 2
            // and a given shell on center 3, loop of shells on 
            // center 4
            for (auto s4=0; s4<basis.size(); ++s4) { 
                auto bf4 = shell2bf[s4]; // index of first bf in shell s4
                auto n4 = basis[s4].size(); // # of func in shell s4

                // Compute eri for first contribution {s1, s2, s3, s4}
                eri_engine.compute(basis[s1], basis[s2], basis[s3], basis[s4]);                  
                
                // Location of computed (shell set) of exchange integrals
                const auto* buf_1234 = buf_vec[0];
        
                // Sum together the eris corresponding to the particular
                // bf on center 1 and the particular bf on center 2
                auto d1 = n2*n3*n4;
                auto d2 = n3*n4;
                auto d3 = n4;
                
                for (auto f3=0; f3<n3; ++f3) {
                  for (auto f4=0; f4<n4; ++f4) {
                    sum12 = sum12 + buf_1234[f1*d1+f2*d2+f3*d3+f4]*C(bf3+f3, j)*C(bf4+f4, b);
                  }
                }
              }
            }
           first_integral(bf1+f1, bf2+f2) = sum12;
          }
        } 
      }
    }           
return first_integral;
}

// Compute second integrals for MP2
Matrix mp2_compute_second(int j, int b, int num_func, BasisSet basis, Matrix C) {

  // Set dimensions of integral matrix
  int n = num_func;

  // Declare integral matrix
  Matrix second_integral(n,n); 
  
  // Create two electron integral engine
  Engine eri_engine(Operator::coulomb,
            basis.max_nprim(),
            basis.max_l()
            );                  

  // Map shell index to basis function index
  auto shell2bf = basis.shell2bf();

  // Point to each computed shell set
  const auto& buf_vec = eri_engine.results();

  // For center 1, loop over shells
  for (auto s1=0; s1<basis.size(); ++s1) { 
    auto bf1 = shell2bf[s1]; // index of first bf in shell s1
    auto n1 = basis[s1].size(); // # of func in shell s1

    // Loop over every bf in shell
    for (auto f1=0; f1<n1; ++f1) {

      // For a given bf in a given shell on center 1,
      // loop over shells on center 2
      for (auto s2=0; s2<basis.size(); ++s2) {  
        auto bf2 = shell2bf[s2]; // index of first bf in shell s2
        auto n2 = basis[s2].size(); // # of func in shell s2

        // Loop over every function in shell
        for (auto f2=0; f2<n2; ++f2) {

          // Initialize sum corresponding to 
          // the bf on center 1 and the bf on center 2
          double sum12 = 0.0;

          // For a given bf on center 1 and a given bf on
          // center 2, loop over shells on center 3
          for (auto s3=0; s3<basis.size(); ++s3) { 
            auto bf3 = shell2bf[s3]; // index of first bf in shell s3
            auto n3 = basis[s3].size(); // # of func in shell s3
           
            // For a given bf on center 1, a given bf on center 2
            // and a given shell on center 3, loop of shells on 
            // center 4
            for (auto s4=0; s4<basis.size(); ++s4) { 
                auto bf4 = shell2bf[s4]; // index of first bf in shell s4
                auto n4 = basis[s4].size(); // # of func in shell s4

                // Compute eri for first contribution {s1, s4, s3, s2}
                eri_engine.compute(basis[s1], basis[s4], basis[s3], basis[s2]);                  
                
                // Location of computed (shell set) of exchange integrals
                const auto* buf_1432 = buf_vec[0];
        
                // Sum together the eris corresponding to the particular
                // bf on center 1 and the particular bf on center 2
                auto d1 = n4*n3*n2;
                auto d2 = n3*n2;
                auto d3 = n2;
                
                for (auto f3=0; f3<n3; ++f3) {
                  for (auto f4=0; f4<n4; ++f4) {
                    sum12 = sum12 + buf_1432[f1*d1+f4*d2+f3*d3+f2]*C(bf3+f3, j)*C(bf4+f4, b);
                  }
                }
              }
            }
           second_integral(bf1+f1, bf2+f2) = sum12;
          }
        } 
      }
    }           
return second_integral;
}


// Compute each individual MP2 term (for each set of indices i, j, a, b)
double comp_mp2_term(int i, int j, int a, int b, int num_func, BasisSet basis, Matrix C, vector<double> eps) {


  // Compute matrices of integrals
  Matrix first_ints = mp2_compute_first(j, b, num_func, basis, C);
  Matrix second_ints = mp2_compute_second(j, b, num_func, basis, C);

  // Contract over i, a index in first integrals
  double full_sum_1 = 0.0;
  for (int mu=0; mu<num_func; ++mu) {
    double part_sum_1 = 0.0;
    for (int nu=0; nu<num_func; ++nu) {
      part_sum_1 += first_ints(mu, nu)*C(nu, a);
    }
    full_sum_1 += part_sum_1*C(mu, i);
  }

  // Contract over i, a index in second integrals
  double full_sum_2 = 0.0;
  for (int mu=0; mu<num_func; ++mu) {
    double part_sum_2 = 0.0;
    for (int nu=0; nu<num_func; ++nu) {
      part_sum_2 += second_ints(mu, nu)*C(nu, a);
    }
    full_sum_2 += part_sum_2*C(mu, i);
  }


  double mp2 = (full_sum_1*(2.0*full_sum_1 - full_sum_2))/(eps[i] + eps[j] - eps[a] - eps[b]);
  return mp2;
}

// Make epsilon vector
vector<double> make_epsilon(int num_func, Matrix F) {
  vector<double> eps(num_func);
  for (int i=0; i<num_func; ++i) {
    eps[i] = F(i,i);
  }
  return eps;
}


//****************************************************************************
// Begin PNOs here
//****************************************************************************


// Compute integrals of form (a mu | b nu) where mu and nu are basis
// functions and a and b are virtual molecular integrals
// Contract over mu and nu to form T^ij matrix



// First step: contract over rho and sigma to make integrals of form
// (a mu | b nu)
Matrix rho_sigma_a_b(BasisSet basis, int num_func, int a, int b, Matrix C) {

  // Create two electron integral engine
  Engine eri_engine(Operator::coulomb,
            basis.max_nprim(),
            basis.max_l()
            );                  

  // Map shell index to basis function index
  auto shell2bf = basis.shell2bf();

  // Point to each computed shell set
  const auto& buf_vec = eri_engine.results();

  // Declare a matrix to hold the integrals of form (ai|bj)
  Matrix munuab(num_func, num_func); // Matrix T^ij

  //double ij_sum = 0.0;


  for (int s1=0; s1<basis.size(); ++s1) { // index i 
    auto bf1 = shell2bf[s1];
    auto n1 = basis[s1].size();

    for (auto f1=0; f1 < n1; ++f1) {

      for (int s2=0; s2<basis.size(); ++s2) { // index j
        auto bf2 = shell2bf[s2];
        auto n2 = basis[s2].size();

        for (auto f2=0; f2<n2; ++f2) {
          double ab_sum = 0.0;

          for (int s3=0; s3<basis.size(); ++s3) { // index a
            auto bf3 = shell2bf[s3];
            auto n3 = basis[s3].size();

            //double ab_sum = 0.0;

            for (int s4=0; s4<basis.size(); ++s4) { // index b
              auto bf4 = shell2bf[s4];
              auto n4 = basis[s4].size();
              
              // Compute eri for (ai | bj)
              eri_engine.compute(basis[s3], basis[s1], basis[s4], basis[s2]);

              // Location of computed shell set
              const auto& buf_3142 = buf_vec[0];

              // Sum together eris relative to specific bfs in s1 and s2
              auto d1 = n1*n4*n2;
              auto d2 = n4*n2;
              auto d3 = n2;

 
              for (auto f3=0; f3<n3; ++f3) {
                for (auto f4=0; f4<n4; ++f4) { 

                  ab_sum += buf_3142[f1*d1 + f2*d2 + f3*d3 + f4]*C(bf3+f3, a)*C(bf4+f4, b);
                }
              }
            }
          }
          munuab(bf1+f1, bf2+f2) = ab_sum;
        }
      }
    }
  }
  return munuab;
}


// Contract over mu and nu to form (a i | b j) integrals
double mu_nu_i_j(int num_func, Matrix munuab, Matrix C, int i, int j) {

  double sum_ij = 0.0;

  for (int mu=0; mu<num_func; ++mu) { // basis functions that make up MO i

    for (int nu=0; nu<num_func; ++nu) { // basis functions that make up MO j

      sum_ij += munuab(mu, nu)*C(mu, i)*C(nu, j);

    }
  }
  return sum_ij;
}

// For a given combination of i and j, form the T^ij matrix
Matrix form_Tij(int num_occ, int num_virt, int num_func, BasisSet basis,
                Matrix C, vector<double> eps, int i, int j) {

  Matrix Tij(num_virt, num_virt);

  for (int a=num_occ; a<num_func; ++a) {

    for (int b=num_occ; b<num_func; ++b) {

      Matrix munuab = rho_sigma_a_b(basis, num_func, a, b, C);
      double sum_ij = mu_nu_i_j(num_func, munuab, C, i, j);
      double element = -sum_ij/(eps[a] + eps[b] - eps[i] - eps[j]);
      Tij(a, b) = element;
    }
  }
  return Tij;
}


// For a given combination of i and j, form the T^ji matrix
Matrix form_Tji() {



int main() {

  libint2::initialize();

  //std::ofstream myfile;
  //myfile.open (NAME);


  // * * * Hartree Fock * * * \\

  // Read in molecular geometry
  std::vector<Atom> atoms = read_geom(COORDS);

  // Create the basis set object
  BasisSet basis = create_bs(BASIS_SET, atoms);

  // Print out the basis set object 
  //copy(begin(basis), end(basis), std::ostream_iterator<Shell>(cout, "\n"));

  // Determine the total number of basis functions in the basis set
  //int num_func  = sum_func(basis);
  int num_func = tot_fun(basis);
  cout << "The total number of basis functions is " << num_func << endl;

  // Determine the total number of occupied orbitals
  int num_occ = count_elec(atoms); 
  cout << "The number of occupied orbitals is " << num_occ << endl;

  // Compute the nuclear attraction energy
  double nuc_energy = nuc_rep(atoms);

  // Form the overlap (S) matrix
  Matrix S = one_elec_compute(basis, num_func, Operator::overlap, atoms);
  //cout << "The overlap (S) matrix: \n\n" << S << "\n" << endl;

  // Diagonalize the S matrix to form the transformation matrix X
  Matrix X = diagonalize_s(S, num_func);
  //cout << "The transformation (X) matrix: \n\n" << X << "\n" << endl;
  
  // Form the kinetic energy (T) matrix
  Matrix T = one_elec_compute(basis, num_func, Operator::kinetic, atoms);
  //cout << "The kinetic energy (T) matrix: \n\n" << T << "\n" << endl;

  // Form the nuclear attraction (V) matrix
  Matrix V = one_elec_compute(basis, num_func, Operator::nuclear, atoms);
  //cout << "The nuclear attraction (V) matrix: \n\n" << V << "\n" << endl;

  // Form the core hamiltonian (H) matrix
  Matrix H = T + V;
  //cout << "The core Hamiltonian (H) matrix: \n\n" << H << "\n" << endl;

  // Form the initial electron density (P) matrix
  Matrix P = make_p(num_func);
  //Matrix P = compute_soad(atoms);
  //Matrix P = Matrix::Zero(num_func, num_func); 
  //Matrix P = Matrix::Identity(num_func, num_func);
  


  //cout << "The initial density (P) matrix: \n\n" << P << "\n" << endl;

  // Form the initial coulomb (J) matrix
  Matrix J = j_eri_compute(basis, P, num_func);
  //cout << "The initial coulomb (J) matrix: \n\n" << J << "\n" << endl;

  // Form the initial exchange (K) matrix 
  Matrix K = k_eri_compute(basis, P, num_func);
  //cout << "The initial exchange (K) matrix: \n\n" << K << "\n" << endl;

  // Form the inital G matrix
  Matrix G = 2*J - K;
  //cout << "The initial G matrix: \n\n" << G << "\n" << endl;

  // Form the initial Fock (F) matrix
  Matrix F = H + G;
  //cout << "The initial Fock (F) matrix: \n\n" << F << "\n" << endl;

  // Form the initial coefficient (C) matrix
  Matrix C(num_func,num_func);

  // Declare matrix to hold epsilon values
  Matrix epsilon(num_func, num_func);

  // Main iterative loop

  // Initialize the frobenius value
  double frob = 1.0;

  // Initialize a count of the iterations
  int iter = 0;  

  // Double convergence value
  double conv = 1e-12;

  // Maximum number of iterations
  int max_iter = 101;

  while (frob > conv && iter < max_iter) {

    // Calculate the transformed Fock (F') matrix
    Matrix Fprime = X.transpose()*F*X;
    
    // Diagonalize F' to obtain C' and epsilon
    Eigen::SelfAdjointEigenSolver<Matrix> es;
    es.compute(Fprime);
    Matrix Cprime = es.eigenvectors();  
    epsilon = es.eigenvalues().asDiagonal();
    

    // Calculated C from C'
    C = X*Cprime;
    //cout << "The C matrix: \n\n" << C << "\n" << endl;

    // Print F'C'
    //cout << "F'*C' is\n" << Fprime*Cprime << endl;

    // Print C'epsilon
    //cout << "C'*epsilon is\n" << Cprime*epsilon << endl;


    Matrix newP(num_func, num_func);
    for (int i=0; i<num_func; ++i) {
      for (int j=0; j<num_func; ++j ) {
        double ij = 0.0;
        for (int k=0; k<num_occ; ++k) {
          ij += C(i, k)*C(j, k);
        }
        newP(i, j) = ij;
      }
    }



    Matrix dif_mat = newP - P;
    frob = dif_mat.norm();
    P = newP;
    J = j_eri_compute(basis, P, num_func);
    K = k_eri_compute(basis, P, num_func);
    F = H + 2*J - K;
    double elec_energy = 0.0;
    for (int i=0; i<num_func; ++i) {
      for (int j=0; j<num_func; ++j) {
        elec_energy = elec_energy + (H(i, j) + F(i, j))*P(i, j);
      }
    }
    iter += 1;
    //cout << "iteration " << iter << "\t" << "frobenius " << frob << "electronic energy " << elec_energy <<  endl;
  }  
  int num_virt = num_func - num_occ;
  cout << "Occ: " << num_occ << endl;
  cout << "Virt: " << num_virt << endl;
  cout << "The number of iterations is " << iter << endl;
  double elec_energy = 0.0;
  for (int i=0; i<num_func; ++i) {
    for (int j=0; j<num_func; ++j) {
      elec_energy = elec_energy + (H(i, j) + F(i, j))*P(i, j);
    }
  }
  //elec_energy = elec_energy/2;

  double EHF = elec_energy + nuc_energy;
  cout << "The Hartree Fock energy is " << EHF << endl;
  cout << "The electronic energy is " << elec_energy << endl;
  cout << "The nuclear repulsion energy is " << nuc_energy << endl;
  //cout << "The matrix of epsilon values is\n " << epsilon << endl;  
  //myfile << "EHF (ang) = " << EHF << endl;
  //myfile << "EHF (bohr) = " << EHF/conv << endl;



  // * * * MP2 * * * \\

  vector<double> eps = make_epsilon(num_func, epsilon);
  
  double mp2_corr = 0.0;

  for (int i=0; i<num_occ; ++i) {
    for (int j=0; j<num_occ; ++j) {
      for (int a=num_occ; a<num_func; ++a) {
        for (int b=num_occ; b<num_func; ++b) {
          double mp2_term = comp_mp2_term(i, j, a, b, num_func, basis, C, eps); 
          mp2_corr += mp2_term;
          cout << i << ", " << j << ", " << a << ", " << b << endl;
        }
      }
    }
  }
  cout << "The MP2 correction to the energy is " << mp2_corr << endl;
  
  //myfile << "MP2 correction (ang) = " << mp2_corr << endl;
  //myfile << "MP2 correction (bohr) = " << mp2_corr/conv << endl;
  //myfile.close();

  libint2::finalize();

  return 0;
}





