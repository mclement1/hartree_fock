#include <libint2.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Eigen>

using namespace std;
using namespace libint2;
using namespace Eigen;

//string MOLECULE = "h2o";
string COORDS = "/Users/mcclement/software/marjory_libint/libint/tests/hartree-fock/h2o.xyz";
string BASIS_SET = "sto-3g";


BasisSet create_bs(string coords, string basis_set) {
    string h2o = coords;
    ifstream input_file(h2o);
    vector<Atom> atoms = read_dotxyz(input_file);

    BasisSet basis(basis_set, atoms);

    return basis;
}

int  main() {

    libint2::initialize();

    BasisSet basis = create_bs(COORDS, BASIS_SET);

    cout << "This is some new text." << endl;

    libint2::finalize();

    return 0;
}
