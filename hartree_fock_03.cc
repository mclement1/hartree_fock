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

/*BasisSet create_bs(std::string coords, std::string basis_set) {
    std::string molecule = coords;
    std::ifstream input_file(molecule);
    std::vector<Atom> atoms = libint2::read_dotxyz(input_file);

    BasisSet basis(basis_set, atoms);

    return basis;
}
*/

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

// Compute 1-electron integrals (nuclear attraction,
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
    
        auto bf1 = shell2bf[s1];
        auto n1 = basis[s1].size();

        for(auto s2=0; s2<=s1; ++s2) {
            
            auto bf2 = shell2bf[s2];
            auto n2 = basis[s2].size();
  
            /*
            cout << "bf1: " << bf1 << ",  ";
            cout << "bf2: " << bf2 << ",  ";
            cout << "n1 x n2: " << n1 << " x " << n2 << endl;
            */
            
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


int  main() {

    libint2::initialize();

    // Read in molecular geometry
    std::vector<Atom> atoms = read_geom(COORDS);
 
    // Create the basis set object
    //BasisSet basis = create_bs(COORDS, BASIS_SET);
    BasisSet basis = create_bs(BASIS_SET, atoms);

    // Print out the basis set object 
    //copy(begin(basis), end(basis), std::ostream_iterator<Shell>(cout, "\n"));

    // Determine the total number of basis functions in the basis set
    int num_func  = sum_func(basis);
    
    //cout << "total number of bf = " << num_func << "\n" << endl;
    
    // Form the overlap (S) matrix
    Matrix s_matrix = one_elec_compute(basis, num_func, Operator::overlap, atoms);
    cout << "The overlap (S) matrix: \n\n" << s_matrix << "\n" << endl;
 
    // Form the kinetic energy (T) matrix
    Matrix t_matrix = one_elec_compute(basis, num_func, Operator::kinetic, atoms);
    cout << "The kinetic energy (T) matrix: \n\n" << t_matrix << "\n" << endl;

    // Form the nuclear attraction (V) matrix
    Matrix v_matrix = one_elec_compute(basis, num_func, Operator::nuclear, atoms);
    cout << "The nuclear attraction (V) matrix: \n\n" << v_matrix << "\n" << endl;

    // Form the core hamiltonian (H) matrix
    Matrix h_matrix = t_matrix + v_matrix;
    cout << "The core Hamiltonian (H) matrix: \n\n" << h_matrix << "\n" << endl;
    libint2::finalize();

    return 0;
}
