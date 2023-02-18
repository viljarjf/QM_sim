#include <Spectra/SymEigsSolver.h>
#include "eigensolver.hpp"
#include "HamiltonianBase.hpp"
#include <iostream>

static inline int max(int a, int b){return a > b ? a : b;}

Eigen::MatrixXd eigen(std::vector<int> N, std::vector<double> L, std::vector<double> stencil, int n_vals, double m){

    HamiltonianBase op(N, L, stencil);

    std::vector<double> V;
    for (int i = 0; i < op.rows(); i++){
        V.push_back(0.5);
    }
    for (int i = op.rows() / 4; i < 3*op.rows() / 4; i++){
        V[i] = 0;
    }
    op.set_potential(V);

    Spectra::SymEigsSolver<HamiltonianBase> eigs(op, n_vals, max(n_vals, N[0] * 3/4));

    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(N[0], n_vals);

    eigs.init();
    int numEigen = eigs.compute(Spectra::SortRule::SmallestAlge, 10000, 1e-30);
    if(eigs.info() == Spectra::CompInfo::Successful)
    {
        res = eigs.eigenvectors();

    }
    else {
        std::cout << "FAILED: " << (int)eigs.info() << "\n";
    }
    
    return res;
}
