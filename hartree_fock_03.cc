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

// Define a Matrix object
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

// Create basis set object
BasisSet create_bs(std::string coords, std::string basis_set) {
    std::string molecule = coords;
    std::ifstream input_file(molecule);
    std::vector<Atom> atoms = libint2::read_dotxyz(input_file);

    BasisSet basis(basis_set, atoms);

    return basis;
}


// Compute overlap integrals and store in Matrix

Matrix s_compute(BasisSet basis) {

    // S matrix dimensions
    //const auto n = basis.size();
    const auto n = 7;
    //cout << "n = " << n << endl;
 
    // Define uninitialized S matrix of appropriate dimensions
    Matrix s_mat(n,n);

    // Create overlap integral engine
    Engine s_engine(Operator::overlap,
                    basis.max_nprim(),
                    basis.max_l()
                    );

        
    // Map shell index to basis function index
    auto shell2bf = basis.shell2bf();

    // Point to each computed shell set
    const auto& s_buf_vec = s_engine.results();

    // Loop over unique pairs of functions
    for (auto s1=0; s1!=basis.size(); ++s1) {
    
        auto bf1 = shell2bf[s1];
        auto n1 = basis[s1].size();

        for(auto s2=0; s2<=s1; ++s2) {
            
            auto bf2 = shell2bf[s2];
            auto n2 = basis[s2].size();
  
            cout << "bf1: " << bf1 << ",  ";
            cout << "bf2: " << bf2 << ",  ";
            cout << "n1 x n2: " << n1 << " x " << n2 << endl;
            
            /*
            cout << "The number of functions in shell s1 is " 
            << shell2bf[s1].size() << "\n";
            cout << "The number of functions in shell s2 is "
            << shell2bf[s2].size() << endl;
            */
            
            // Compute overlap integral
            s_engine.compute(basis[s1], basis[s2]);
    
            // Store overlap integral value in uninitialized Matrix
            //auto s_shellset = s_buf_vec[0];
            Eigen::Map<const Matrix> s_buf_mat(s_buf_vec[0], n1, n2); 
            s_mat.block(bf1, bf2, n1, n2 ) = s_buf_mat;
            
            if(s1!=s2)
                s_mat.block(bf2, bf1, n2, n1) = s_buf_mat.transpose();


       }
    }
    return s_mat;
}





int  main() {

    libint2::initialize();

    BasisSet basis = create_bs(COORDS, BASIS_SET);

            
    // Print out the basis set object 
    copy(begin(basis), end(basis), std::ostream_iterator<Shell>(cout, "\n"));

    Matrix s_matrix = s_compute(basis);

    cout << "The overlap (S) matrix: \n" << s_matrix << endl;
    
    libint2::finalize();

    return 0;
}
