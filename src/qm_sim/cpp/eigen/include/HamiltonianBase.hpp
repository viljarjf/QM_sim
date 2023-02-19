#include <vector>
#include <Eigen/Core>

class HamiltonianBase
{
    std::vector<int> N;
    std::vector<double> L;
    int ndim;

    double m;

    std::vector<double> stencil;

    std::vector<double> V;

    // -hbar^2 / 2m dx^2 for each dimension, in reduced units
    std::vector<double> h0;

    // Vector from 0 to n where i_vec[i] = i
    std::vector<int> i_vec;

    template <typename arr_in, typename arr_out>
    void matrix_multiply(arr_in &x_in, arr_out &y_out) const;

public:
    using Scalar = double;  // A typedef named "Scalar" is required

    /**
     * @brief Construct a new Hamiltonian Base object
     * 
     * @param N discretisation count for each dimension
     * @param L System size in each dimension, units of nanometer
     * @param stencil Laplacian discretisation scheme
     * @param m particle mass, units of electron mass
     */
    HamiltonianBase(std::vector<int> N, std::vector<double> L, std::vector<double> stencil, double m = 1.0);

    const int rows() const {
        int out = 1;
        for (int i = 0; i < this->ndim; i++){
            out *= this->N[i];
        }
        return out;
    };
    const int cols() const { return this->rows(); };

    // Set the potential of the system.
    // units eV
    void set_potential(std::vector<double> V){this->V = V;};
    
    void perform_op(const double *x_in, double *y_out) const;

    Eigen::VectorXd as_operator(Eigen::VectorXd x_in);
};
