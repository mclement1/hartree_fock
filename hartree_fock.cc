#include <libint2.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <Eigen/Eigen>


using namespace std;
using namespace libint2;
using namespace Eigen;


int main() {

    // Initialize libint

    libint2::initialize();

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

    // Map shell index to basis function index

    auto shell2bf = obs.shell2bf();
   
 
    // Compute overlap integrals

    const auto& s_buf_vec = s_engine.results();

    for(auto s1=0; s1!=obs.size(); ++s1) {
        for(auto s2=0; s2!=obs.size(); ++s2) {
        
            s_engine.compute(obs[s1], obs[s2]);
            auto s_shellset = s_buf_vec[0]; // points to the location of the first integral
            if (s_shellset == nullptr)
                continue;

        }
    }    
    

    // Compute kinetic energy integrals

    const auto& t_buf_vec = t_engine.results();

    for(auto s1=0; s1!=obs.size(); ++s1) {
        
        t_engine.compute(obs[s1], obs[s1]);
        auto t_shellset = t_buf_vec[0]; // points to the location of the first integral
        if (t_shellset == nullptr)
        continue;
    }   
    
    // Compute nuclear attraction integrals

    const auto& v_buf_vec = v_engine.results();

    for(auto s1=0; s1!=obs.size(); ++s1) {

        v_engine.compute(obs[s1], obs[s1]);
        auto v_shellset = v_buf_vec[0]; // points to the location of the first integral
        if (v_shellset == nullptr)
        continue;
    }   


    // Compute two-body integrals

    /* const auto& eri_buf_vec = eri_engine.results();

    for(auto s1=0; s1!=obs.size(); ++s1) {
        for(auto s2=0; s2!=obs.size(); ++s2) {
            for(auto s3=0; s3!=obs.size(); ++s3) {
                for(auto s4=0; s4!=obs.size(); ++s4) {

                    s_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
                    auto eri_shellset = eri_buf_vec[0]; // points to the location of the first integral
                    if (eri_shellset == nullptr)
                        continue;
                }        
            }
        }
    }    
 */
    // Form core-Hamiltonian matrix, H

    // Form overlap matrix, S

    // Map<MatrixXd> s_mat(s_shellset, obs.size(), obs.size();

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

    libint2::finalize();

    // prints some text
    cout << "This is some text." << endl;

    return 0;

}
