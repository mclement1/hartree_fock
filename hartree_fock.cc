#include <libint2.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>

using namespace std;
using namespace libint2;



int main() {

    // Initialize libint

    libint2::initialize();

    // Initialize eigen

    // Read in geometry and create basis set object

    string h2o = "/Users/mcclement/software/marjory_libint/libint/tests/hartree-fock/h2o.xyz"; 
    ifstream input_file(h2o);
    vector<Atom> atoms = read_dotxyz(input_file);

    BasisSet obs("sto-3g", atoms);

    // Create overlap integral engine (overlap)

    Engine s_engine(Operator::overlap,
                    obs.max_nprim(),
                    obs.max_l()
                    );
                
    // Create kinetic energy integral engine (kinetic)

    Engine t_engine(Operator::kinetic,
                    obs.max_nprim(),
                    obs.max_l()
                    );

    // Create nuclear attraction integral engine (nuclear)

    Engine v_engine(Operator::nuclear,
                    obs.max_nprim(),
                    obs.max_l()
                    );
    
    v_engine.set_params(make_point_charges(atoms));

    // Create two-body electron repulsion integral engine (coulomb)

    Engine eri_engine(Operator::coulomb,
                      obs.max_nprim(),
                      obs.max_l()
                      );


    // Compute overlap integrals

    // Compute kinetic energy integrals

    // Compute nuclear attraction integrals

    // Compute two-body integrals

    // Form core-Hamiltonian matrix, H

    // Form overlap matrix, S

    // Diagonalize overlap matrix, S

    // Form transformation matrix, X

    // Beginning of iterative loop

    // Form guess density matrix, P

    // Form the matrix G

    // Form the Fock matrix, F

    // Calculate the transformed Fock matrix F' 

    // Diagonalize F' to obtain C' and epsilon

    // Calculate C from C'

    // Form a new density matrix P from C

    // Feed new P back into iterative loop

    // Leave loop when convergence has been reached
    
    // Finalize libint

    // Finalize eigen


    // prints some text
    cout << "This is some text." << endl;

    return 0;

}
