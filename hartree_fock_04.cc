// C++ headers
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// Libint gaussian integrals library
#include <libint2.hpp>

using std::cout;
using std::cerr;
using std::endl;

using libint2::BasisSet;
using libint2::Atom;
using libint2::Shell;
using libint2::Engine;
using libint2::Operator;

// Location of geometry file and basis set file
std::string COORDS = "/Users/mcclement/practice/hartree_fock/h2o.xyz";
std::string BASIS_SET = "sto-3g";

// Define a Matrix type 
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

// Define an atom type
//typedef std::vector<Atom> atom;

// Read in molecular coordinates
std::vector<Atom> read_geom(std::string coords) {
  std::string molecule = coords;
  std::ifstream input_file(molecule);
  std::vector<Atom> atoms = libint2::read_dotxyz(input_file);
  return atoms;
}

// Create basis set object
BasisSet create_bs(std::string basis_set, std::vector<Atom> atoms) {
    
  BasisSet basis(basis_set, atoms);
  return basis;
}

//BasisSet create_bs(std::string coords, std::string basis_set) {
//    std::string molecule = coords;
//    std::ifstream input_file(molecule);
//    std::vector<Atom> atoms = libint2::read_dotxyz(input_file);
//
//    BasisSet basis(basis_set, atoms);
//
//    return basis;
//}
//

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

// Compute the nuclear attraction energy

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

   
   // cout << "bf1: " << bf1 << ",  ";
   // cout << "bf2: " << bf2 << ",  ";
   // cout << "n1 x n2: " << n1 << " x " << n2 << endl;
    
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

////Compute the electron density matrix, P
////for initial purposes, use Pij = 0 
//
//  Matrix compute_density(Vector coeff, int num_func) {
//
//  Determine dimensions of density matrix
//  int n = num_func;
//
//  //Declare the denisty (P) matrix
//  //and initialize with all zeros
//  Matrix density(n, n) = Matrix::Zero;
//
//  return density;
//
//}

 

// Compute two-electron integrals (electron repulsion integrals)
// and form G matrix 

Matrix eri_compute(BasisSet basis, Matrix density_mat, int num_func) {

  // Set dimensions of J and K matrices
  int n = num_func;

  // Declare coulomb (J) matrix
  Matrix coulomb(n,n); 
  
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

  // Sum the functions in each shell
  const auto nf = sum_func(basis);

  // Declare a vector to hold the eri values
  //std::vector<double> vals(nf*nf*nf*nf);


// Begin a running sum to determine where in
// the vector each new shell set should start 
  //int index = 0;

  // Loop over every combination of shells
  for (auto s1=0; s1<basis.size(); ++s1) { 
    auto bf1 = shell2bf[s1]; // index of first bf in shell s1
    auto n1 = basis[s1].size(); // # of func in shell s1

    for (auto s2=0; s2<basis.size(); ++s2) {  
      auto bf2 = shell2bf[s2]; // index of first bf in shell s2
      auto n2 = basis[s2].size(); // # of func in shell s2

      double J12 = 0.0;
      double K12 = 0.0;


      for (auto s3=0; s3<basis.size(); ++s3) { 
        auto bf3 = shell2bf[s3]; // index of first bf in shell s3
        auto n3 = basis[s3].size(); // # of func in shell s3
       
        for (auto s4=0; s4<basis.size(); ++s4) { 
            auto bf4 = shell2bf[s4]; // index of first bf in shell s4
            auto n4 = basis[s4].size(); // # of func in shell s4

            // Compute eri for coulomb contribution {s1, s2, s3, s4}
            eri_engine.compute(basis[s1], basis[s2], basis[s3], basis[s4]);                  
            //cout << bf1 << "," << bf2 << "," << bf3 << "," << bf4 << endl;
            
            // Location of computed (shell set) of coulomb integrals
            const auto* buf_1234 = buf_vec[0];
           
            // Store eri in vector
            for (auto f1=0; f1<n1; ++f1) {
              auto d1 = n2*n3*n4; // # of basis funcs per func in s1

              for (auto f2=0; f2<n2; ++f2) {
                auto d2 = n3*n4; // # of basis funcs per func in s2

                for (auto f3=0; f3<n3; ++f3) {
                  auto d3 = n4; // # of basis funcs per func in s3
                  
                  for (auto f4=0; f4<n4; ++f4) {
                    J12 = J12 + buf_1234[f1*d1+f2*d2+f3*d3+f4];


                  //cout << eri_shellset[f1*d1 + f2*d2 + f3*d3 + f4] << endl;
                  //vals[index+f1*d1+f2*d2+f3*d3+f4] = eri_shellset[f1*d1+f2*d2+f3*d3+f4];
                  //vals[()] = eri_shellset[f1*d1+f2*d2+f3*d3+f4]
                  }                                  
                }
              }
            }  

            // Compute eri for exchange contribution {s1, s4, s3, s2}
            eri_engine.compute(basis[s1], basis[s4], basis[s3], basis[s2]);

            // Location of computed shell set of exchange integrals
            const auto* buf_1432 = buf_vec[0];
                                                                    
            //index = index + n1*n2*n3*n4;
            //cout << "index = " << index << endl;
          } 
        }
      }
    }

  
// cout << "\ns1, s2, s3, s4: " << s1 << ", " << s2 << ", " << s3 << ", " << s4 << endl;
// cout << "bf1, bf2, bf3, bf4: " << bf1 << ", " << bf2 << ", " << bf3 << ", " << bf4 << endl; 
// cout << "n1, n2, n3, n4: " << n1 << ", " << n2 << ", " << n3 << ", " << n4 << endl;
// cout << "index: " << index << endl;
 
return vals;

}


int  main() {

  libint2::initialize();

  // Read in molecular geometry
  std::vector<Atom> atoms = read_geom(COORDS);

  // Create the basis set object
  BasisSet basis = create_bs(BASIS_SET, atoms);

  // Print out the basis set object 
  //copy(begin(basis), end(basis), std::ostream_iterator<Shell>(cout, "\n"));

  // Determine the total number of basis functions in the basis set
  int num_func  = sum_func(basis);

  //cout << "total number of bf = " << num_func << "\n" << endl;

  // Compute the nuclear attraction energy

  // Form the overlap (S) matrix
  Matrix S = one_elec_compute(basis, num_func, Operator::overlap, atoms);
  cout << "The overlap (S) matrix: \n\n" << S << "\n" << endl;

  // Form the kinetic energy (T) matrix
  Matrix T = one_elec_compute(basis, num_func, Operator::kinetic, atoms);
  cout << "The kinetic energy (T) matrix: \n\n" << T << "\n" << endl;

  // Form the nuclear attraction (V) matrix
  Matrix V = one_elec_compute(basis, num_func, Operator::nuclear, atoms);
  cout << "The nuclear attraction (V) matrix: \n\n" << V << "\n" << endl;

  // Form the core hamiltonian (H) matrix
  Matrix H = T + V;
  cout << "The core Hamiltonian (H) matrix: \n\n" << H << "\n" << endl;

  // Form the electron density (P) matrix
  Matrix P = Matrix::Zero(num_func, num_func);
  cout << "The initial density (P) matrix: \n\n" << P << "\n" << endl;

  // Form the coulomb (J) matrix

  // Form the exchange (K) matrix 

  // Form the Fock (F) matrix
  // Matrix F = H + 2*J - K;


  //std::vector<double> vals = eri_compute(basis);
  //cout << vals[0] << endl;
  //  
  //auto shells = basis.size();
  //cout << "total # of shells: " << shells << endl;
  //auto nf = sum_func(basis);
  //cout << "total # of bf: " << nf << endl;
  //auto size = vals.size();
  //cout << "total # of eris: " << size << endl;
  //cout << "nf*nf*nf*nf << endl;

  libint2::finalize();


  return 0;
}
