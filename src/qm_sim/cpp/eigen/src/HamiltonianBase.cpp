#include "HamiltonianBase.hpp"
#include <iostream>
#include <execution>


HamiltonianBase::HamiltonianBase(std::vector<int> N, std::vector<double> L, std::vector<double> stencil, double m) : 
N{N}, L{L}, stencil{stencil}, m{m} {
    this->ndim = L.size();
    for (int i = 0; i < this->rows(); i++){
        this->V.push_back(0.0);
    }
    double dx;
    for(int dim = 0; dim < ndim; dim++){
        dx = this->L[dim] / this->N[dim];
        // - hbar^2 / 2m dx^2 in units of electron mass, eV, and nm, 
        // according to wolfram alpha
        this->h0.push_back(
            -0.0380998212 / (this->m * dx * dx)
        );
    }
    for(int i = 0; i < this->rows(); i++){this->i_vec.push_back(i);}

}

template <typename arr_in, typename arr_out>
void HamiltonianBase::matrix_multiply( arr_in &x_in, arr_out &y_out) const {
    int n = this->rows();
    for (int i = 0; i < n; i++){
        y_out[i] = this->V[i] * x_in[i];
    }

    double h0_dim;
    int offset = 1;
    for (int dim = 0; dim < this->ndim; dim++){

        // 1 / dx^2
        h0_dim = this->h0[dim];

        auto stencil = this->stencil;
        // y_i = x_i (*) stencil
        // stencil usually has more terms, so it can be e.g.
        // y_i = stencil[1]*x_i-1 + stencil[0]*x_i + stencil[1]*x_i+1
        std::for_each(
            std::execution::par,
            std::begin(this->i_vec),
            std::end(this->i_vec),
            [=,  &y_out](int i)
            {
            double stencil_sum, stencil_el;
            stencil_sum = x_in[i] * stencil[0];
            for (int j = 1; j < stencil.size(); j++){
                stencil_el = stencil[j];
                // account for current dimension
                j *= offset;
                // range-check the x-vector
                if ((i + j) < n){
                    stencil_sum += x_in[i + j] * stencil_el;
                }
                if ((i - j) >= 0){
                    stencil_sum += x_in[i - j] * stencil_el;
                }
            }
            y_out[i] += h0_dim * stencil_sum;
        }); // end of parfor
        offset *= this->N[dim];
    }
}

void HamiltonianBase::perform_op(const double *x_in, double *y_out) const {
    this->matrix_multiply<const double *, double *>(x_in, y_out);
}

Eigen::VectorXd HamiltonianBase::as_operator(Eigen::VectorXd x_in){
    Eigen::VectorXd y_out {x_in.size()};
    this->matrix_multiply(x_in, y_out);
    return y_out;
}
