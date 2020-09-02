#include "ConePlacer.h"

ConePlacer::ConePlacer(ManifoldSurfaceMesh& mesh_, VertexPositionGeometry& geo_)
    : mesh(mesh_), geo(geo_) {
    vIdx = mesh.getVertexIndices();

    Vector<bool> isInterior(mesh.nVertices());
    for (Vertex v : mesh.vertices()) {
        isInterior(vIdx[v]) = !v.isBoundary();
    }

    geo.requireCotanLaplacian();
    SparseMatrix<double> L = geo.cotanLaplacian;
    BlockDecompositionResult<double> Ldecomp =
        blockDecomposeSquare(L, isInterior);

    Lii = Ldecomp.AA;
    Lib = Ldecomp.AB;

    geo.requireVertexLumpedMassMatrix();
    SparseMatrix<double> M = geo.vertexLumpedMassMatrix;
    BlockDecompositionResult<double> Mdecomp =
        blockDecomposeSquare(M, isInterior);
    Mii = Mdecomp.AA;

    std::cout << "L rows: " << L.rows() << "\tLii rows: " << Lii.rows()
              << "\t M rows: " << Mii.rows()
              << "\tnInteriorVertices: " << mesh.nInteriorVertices() << endl;

    geo.requireVertexGaussianCurvatures();
    VertexData<double> Omega = geo.vertexGaussianCurvatures;
    Omegaii                  = Vector<double>(mesh.nInteriorVertices());

    std::vector<Eigen::Triplet<double>> ET, RT, WeT;

    size_t iV = 0;
    for (Vertex v : mesh.vertices()) {
        if (!v.isBoundary()) {
            Omegaii(iV++) = Omega[v];
        }
    }
}

std::array<VertexData<double>, 3>
ConePlacer::computeOptimalMeasure(double lambda, size_t regularizedSteps) {
    if (verbose) cout << "Computing Optimal Cones with λ = " << lambda << endl;

    // Normalize lambda by surface area?
    double surfaceArea = 0;
    geo.requireFaceAreas();
    for (Face f : mesh.faces()) surfaceArea += geo.faceAreas[f];
    lambda /= surfaceArea;

    if (verbose) cout << "\t\tafter normalization, λ = " << lambda << endl;

    Vector<double> u   = Vector<double>::Constant(mesh.nInteriorVertices(), 0);
    Vector<double> phi = Vector<double>::Constant(mesh.nInteriorVertices(), 0);
    double gamma       = 1;

    if (verbose) cout << "Beginning Regularized Solves" << endl;

    for (size_t iS = 0; iS < regularizedSteps; ++iS) {
        std::tie(u, phi) = computeRegularizedMeasure(u, phi, lambda, gamma);
        gamma /= 10;
        if (verbose) cout << "\tFinished iteration " << iS << endl;
    }

    if (verbose) cout << "Beginning Exact Solve" << endl;
    return computeOptimalMeasure(u, phi, lambda);
}

VertexData<double> ConePlacer::contractClusters(const VertexData<double>& x) {
    double tol = 1e-12;
    VertexData<bool> visited(mesh, false);
    VertexData<double> contracted(mesh, 0);

    auto contractCluster = [&](Vertex v) {
        if (visited[v] || abs(x[v]) < tol) {
            visited[v] = true;
            return;
        } else if (v.isBoundary()) {
            contracted[v] = x[v];
        }

        Vertex mode       = v;
        double clusterSum = 0;

        std::deque<Vertex> clusterVertices;
        clusterVertices.push_back(v);
        visited[v] = true;

        while (!clusterVertices.empty()) {
            Vertex v = clusterVertices.front();
            clusterVertices.pop_front();

            if (abs(x[v]) >= tol) {
                clusterSum += x[v];
                if (abs(x[v]) > abs(x[mode])) mode = v;
                for (Vertex w : v.adjacentVertices()) {
                    if (!visited[w] && !w.isBoundary()) {
                        clusterVertices.push_back(w);
                        visited[w] = true;
                    }
                }
            }
        }

        contracted[mode] = clusterSum;
    };

    for (Vertex v : mesh.vertices()) {
        contractCluster(v);
    }

    return contracted;
}

VertexData<double> ConePlacer::pruneSmallCones(const VertexData<double>& mu,
                                               double threshold) {
    return pruneSmallCones(mu, threshold, threshold);
}

VertexData<double> ConePlacer::pruneSmallCones(VertexData<double> mu,
                                               double positiveThreshold,
                                               double negativeThreshold) {
    double maxCone = 0;
    double minCone = 0;
    for (Vertex v : mesh.vertices()) {
        maxCone = fmax(maxCone, mu[v]);
        minCone = fmin(minCone, mu[v]);
    }

    double targetConeSum = 2 * M_PI * mesh.eulerCharacteristic();
    double coneSum       = 0;
    for (Vertex v : mesh.vertices()) {
        if (mu[v] > 0) {
            if (mu[v] < positiveThreshold * maxCone) {
                mu[v] = 0;
            } else {
                coneSum += mu[v];
            }
        } else if (mu[v] < 0) {
            if (mu[v] > negativeThreshold * minCone) {
                mu[v] = 0;
            } else {
                coneSum += mu[v];
            }
        }
    }

    for (Vertex v : mesh.vertices()) {
        mu[v] *= targetConeSum / coneSum;
    }

    return mu;
}

void ConePlacer::setVerbose(bool verb) { verbose = verb; }

std::array<Vector<double>, 2>
ConePlacer::computeRegularizedMeasure(Vector<double> u, Vector<double> phi,
                                      double lambda, double gamma) {
    size_t n             = mesh.nInteriorVertices();
    double residualNorm2 = 100;

    Vector<double> ones = Vector<double>::Constant(mesh.nInteriorVertices(), 1);

    size_t iter = 0;
    while (residualNorm2 > 1e-5 && iter++ < 50) {
        Vector<double> b = -regularizedResidual(u, phi, lambda, gamma);

        SparseMatrix<double> DF = computeRegularizedDF(u, phi, lambda, gamma);
        Vector<double> stackedStep = solveSquare(DF, b);

        // cerr << "solve residual: " << (DF * stackedStep - b).norm() << endl;

        std::vector<Vector<double>> step = unstackVectors(stackedStep, {n, n});

        u += step[0];
        // cerr << "total step norm: " << stackedStep.norm()
        //      << "\tu step norm: " << step[0].norm()
        //      << "\t rhs norm: " << b.norm() << endl;

        phi += step[1];
        // u   = projectOutConstant(u);
        // phi = computePhi(u);
        // phi = projectOutConstant(phi);

        // if (u.dot(Mii * ones) > 1e-8) {
        //     cerr << "projectOutOnes(u) failed" << endl;
        // }
        // if (phi.dot(Mii * ones) > 1e-8) {
        //     cerr << "projectOutOnes(phi) failed" << endl;
        // }

        // if ((Lii.transpose() * phi - Mii.transpose() * u).norm() > 1e-8) {
        //     cerr << "projectOutOnes broke poisson equation" << endl;
        //     // exit(1);
        // }

        residualNorm2 = step[0].squaredNorm() + step[1].squaredNorm();
        if (verbose)
            cout << "\t\t" << iter << "\t| residual: " << residualNorm2
                 << "\t| u norm: " << u.lpNorm<1>()
                 << "\t| phi norm: " << phi.lpNorm<1>() << endl;
        // exit(1);
    }

    return {u, phi};
}

std::array<VertexData<double>, 3>
ConePlacer::computeOptimalMeasure(Vector<double> u, Vector<double> phi,
                                  double lambda) {
    Vector<double> mu    = P(phi, lambda);
    size_t n             = mesh.nInteriorVertices();
    double residualNorm2 = residual(u, phi, mu, lambda).lpNorm<2>();

    if (verbose)
        cout << "\t initial residual: "
             << residual(u, phi, mu, lambda).lpNorm<2>()
             << "\t mu norm: " << mu.lpNorm<1>() << endl;

    size_t iter = 0;
    while (residualNorm2 > 1e-12 && iter++ < 1000) {

        Vector<double> b = residual(u, phi, mu, lambda);

        SparseMatrix<double> DF = computeDF(u, phi, mu, lambda);

        SquareSolver<double> solver(DF);

        std::vector<Vector<double>> step =
            unstackVectors(solver.solve(-b), {n, n, n});

        u += step[0];
        phi += step[1];
        mu += step[2];
        // u   = projectOutConstant(u);
        // phi = projectOutConstant(phi);
        // u   = computeU(mu);
        // phi = computePhi(u);

        // TODO: should I normalize mu?
        // mu = normalizeMuSum(mu);

        residualNorm2 = step[0].squaredNorm() + step[1].squaredNorm() +
                        step[2].squaredNorm();
        if (verbose)
            cout << "\t" << iter << "\t| residual: " << residualNorm2
                 << "\t u norm: " << u.lpNorm<1>()
                 << "\t phi norm: " << phi.lpNorm<1>()
                 << "\t mu norm: " << mu.lpNorm<1>()
                 << "\t step2 norm: " << step[2].lpNorm<1>() << endl;
    }

    Vector<double> r = residual(u, phi, mu, lambda);
    if (verbose) {
        cout << "\t final residual: " << r.lpNorm<2>() << endl;

        Vector<double> ones = Vector<double>::Constant(u.rows(), 1);
        cout << "\t 1^T M u: " << ones.dot(Mii * u) << endl;
        cout << "\t 1^T M phi: " << ones.dot(Mii * phi) << endl;
        cout << "\t 1^T (Omega - mu): " << ones.dot(Omegaii - mu) << endl;
    }

    checkSubdifferential(mu, phi, lambda);
    // checkPhiIsEnergyGradient(mu, phi, lambda);

    auto assignInteriorVertices = [&](const Vector<double>& vec) {
        VertexData<double> data(mesh, 0);
        size_t iV = 0;
        for (Vertex v : mesh.vertices()) {
            if (!v.isBoundary()) {
                data[v] = vec(iV++);
            }
        }
        return data;
    };

    VertexData<double> uData   = assignInteriorVertices(u);
    VertexData<double> phiData = assignInteriorVertices(phi);

    Vector<double> fullU   = uData.toVector();
    SparseMatrix<double> L = geo.cotanLaplacian;
    Vector<double> Omega   = geo.vertexGaussianCurvatures.toVector();

    VertexData<double> muData(mesh, Omega - L * fullU);

    if (verbose) {
        Vector<double> uErr = Lii * u - Omegaii + mu;
        cerr << "u err: " << uErr.norm() << endl;

        Vector<double> computedU = computeU(mu);
        cerr << "computed u residual: "
             << (Lii * computedU - Omegaii + mu).norm() << endl;
        cerr << "solved u err (compared to solution u): "
             << (computedU - u).norm() << endl;
        cerr << "Lii * solved u err: " << (Lii * (computedU - u)).norm()
             << endl;
        for (size_t iV = 0; iV < fmin(10, computedU.size()); ++iV)
            cerr << computedU(iV) << "\t" << u(iV) << "\t"
                 << computedU(iV) - u(iV) << endl;

        Vector<double> ones = Vector<double>::Constant(uErr.rows(), 1);

        cerr << "u . ones : " << u.dot(ones)
             << "\t computedU . ones: " << computedU.dot(ones) << endl;

        double netConeAngle = muData.toVector().sum();
        cerr << "net cone angle: " << netConeAngle
             << "\t 2 pi chi: " << 2 * M_PI * mesh.eulerCharacteristic()
             << endl;
        // SparseMatrix<double> DF = computeDF(u, phi, mu, lambda);
        // Eigen::SparseQR<SparseMatrix<double>, Eigen::COLAMDOrdering<int>> lu;
        // lu.compute(DF);
        // cerr << "DF nullity = " << DF.rows() - lu.rank() << endl;
        // lu.compute(Lii);
        // cerr << "Lii nullity = " << Lii.rows() - lu.rank() << endl;
        // cerr << "Lii * ones norm:" << (Lii * ones).norm() << endl;
    }


    return {uData, phiData, muData};
}

Vector<double> ConePlacer::regularizedResidual(const Vector<double>& u,
                                               const Vector<double>& phi,
                                               double lambda, double gamma) {
    Vector<double> r0 = Lii * u - Omegaii + P(phi, lambda) / gamma;
    Vector<double> r1 = Lii.transpose() * phi - Mii * u;
    return stackVectors({r0, r1});
}

Vector<double> ConePlacer::residual(const Vector<double>& u,
                                    const Vector<double>& phi,
                                    const Vector<double>& mu, double lambda) {
    Vector<double> r0 = Lii * u - Omegaii + mu;
    Vector<double> r1 = Lii.transpose() * phi - Mii * u;
    Vector<double> r2 = mu - P(Vector<double>(phi + mu), lambda);
    return stackVectors({r0, r1, r2});
}

SparseMatrix<double> ConePlacer::computeDF(const Vector<double>& u,
                                           const Vector<double>& phi,
                                           const Vector<double>& mu,
                                           double lambda) {
    SparseMatrix<double> zeros(mesh.nInteriorVertices(),
                               mesh.nInteriorVertices());
    SparseMatrix<double> id(mesh.nInteriorVertices(), mesh.nInteriorVertices());
    id.setIdentity();
    SparseMatrix<double> Dphi   = D(phi, lambda);
    SparseMatrix<double> Dphimu = D(phi + mu, lambda);
    // clang-format off
    SparseMatrix<double> top = horizontalStack<double>({Lii,   zeros,           id});
    SparseMatrix<double> mid = horizontalStack<double>({- Mii, Lii.transpose(), zeros});
    SparseMatrix<double> bot = horizontalStack<double>({zeros, -Dphimu,         id - Dphimu});
    // clang-format on

    return verticalStack<double>({top, mid, bot});
}

SparseMatrix<double> ConePlacer::computeRegularizedDF(const Vector<double>& u,
                                                      const Vector<double>& phi,
                                                      double lambda,
                                                      double gamma) {
    SparseMatrix<double> topRight = D(phi, lambda) / gamma;

    SparseMatrix<double> top = horizontalStack<double>({Lii, topRight});
    SparseMatrix<double> bot = horizontalStack<double>({-Mii, Lii.transpose()});

    return verticalStack<double>({top, bot});
}

double ConePlacer::muSum(const Vector<double>& mu) {
    double sum = 0;
    for (size_t iV = 0; iV < (size_t)mu.rows(); ++iV) sum += mu(iV);
    return sum;
}

Vector<double> ConePlacer::normalizeMuSum(const Vector<double>& mu) {
    double targetSum = 2 * M_PI * mesh.eulerCharacteristic();
    return targetSum / muSum(mu) * mu;
}

Vector<double> ConePlacer::P(Vector<double> x, double lambda) {
    for (size_t iV = 0; iV < (size_t)x.rows(); ++iV) {
        x(iV) = fmax(0, x(iV) - lambda) + fmin(0, x(iV) + lambda);
    }
    return x;
}


SparseMatrix<double> ConePlacer::D(const Vector<double>& x, double lambda) {
    std::vector<Eigen::Triplet<double>> T;

    std::vector<double> diag = Dvec(x, lambda);

    for (size_t iE = 0; iE < diag.size(); ++iE) {
        if (abs(diag[iE]) > 1e-12) T.emplace_back(iE, iE, diag[iE]);
    }

    SparseMatrix<double> M(diag.size(), diag.size());
    M.setFromTriplets(std::begin(T), std::end(T));

    return M;
}

std::vector<double> ConePlacer::Dvec(const Vector<double>& x, double lambda) {
    std::vector<double> result;
    for (size_t iV = 0; iV < (size_t)x.rows(); ++iV) {
        if (abs(x(iV)) > lambda) {
            result.push_back(1);
        } else {
            result.push_back(0);
        }
    }
    return result;
}

bool ConePlacer::checkSubdifferential(const Vector<double>& mu,
                                      const Vector<double>& phi,
                                      double lambda) {
    // Check F-R subdifferential
    Vector<double> psi(mesh.nInteriorVertices());
    for (size_t iV = 0; iV < mesh.nInteriorVertices(); ++iV) {
        psi(iV) = (mu(iV) > 0) ? lambda : -lambda;
    }

    // Subdifferential condition says for all psi, mu(psi) <= mu(phi)
    double err = mu.dot(psi) - mu.dot(phi);

    if (err > 1e-8) {
        cerr << "Not a subdifferential after all" << endl;
    } else {
        cerr << "Subdifferential okay" << endl;
    }

    // Check abstract subdifferential
    double Rmu = lambda * mu.lpNorm<1>();

    err = abs(Rmu - mu.dot(phi));

    if (err > 1e-8) {
        cerr << "Not an abstract subdifferential after all" << endl;
    } else {
        cerr << "Abstract subdifferential okay" << endl;
    }

    // Check discrete subdifferential
    // Minimizers of e(μ) + |μ|_1 satisfy
    // φ \in \partial λ|μ|_1 (since φ = -grad e),
    // which means that φ_i = λ sgn(μ_i) for μ_i nonzero
    double worstErr = 0;
    for (size_t iV = 0; iV < (size_t)mu.rows(); ++iV) {
        if (abs(mu(iV)) > 1e-12) {
            err      = phi(iV) - std::copysign(lambda, mu(iV));
            worstErr = fmax(abs(err), worstErr);
        }
    }
    if (worstErr > 1e-8) {
        cerr << "Discrete subdifferential fails" << endl;
    } else {
        cerr << "Discrete subdifferential okay" << endl;
    }

    return true;
}

double ConePlacer::checkPhiIsEnergyGradient(const Vector<double>& mu,
                                            const Vector<double>& phi,
                                            double lambda, double epsilon) {
    double worstErr = 0;
    double L2Err    = 0;
    double muEnergy = computeDistortionEnergy(mu, lambda);
    if (abs(muEnergy - Lagrangian(mu, computeU(mu), phi)) > 1e-8) {
        cerr << "Error in Lagrangian" << endl;
    } else {
        cerr << "Lagrangian Okay" << endl;
    }

    Vector<double> standardizedPhi = projectOutConstant(phi);

    Vector<double> finiteDifferenceGradient(mu.rows());
    for (size_t iV = 0; iV < (size_t)mu.rows(); ++iV) {
        Vector<double> dmu = Vector<double>::Zero(mu.rows());
        dmu(iV)            = epsilon;

        double perturbedEnergy  = computeDistortionEnergy(mu + dmu, lambda);
        double finiteDifference = (perturbedEnergy - muEnergy) / epsilon;
        finiteDifferenceGradient(iV) = finiteDifference;

        double err = abs(phi(iV) + finiteDifference);

        if (iV < 10 && verbose && err > 1e-4) {
            cout << "finiteDifference: " << finiteDifference
                 << "\tphi: " << phi(iV)
                 << "\tstandardized phi: " << standardizedPhi(iV) << endl;
        }

        worstErr = fmax(worstErr, err);

        L2Err += err * err;
    }

    L2Err = sqrt(L2Err / mu.rows());
    if (verbose) {
        Vector<double> ones = Vector<double>::Constant(mu.rows(), 1);
        cout << "worst phi err: " << worstErr << "\t l2 err: " << L2Err << endl;
        cout << "1^T M * finite difference grad / norm = "
             << ones.dot(Mii * finiteDifferenceGradient) /
                    finiteDifferenceGradient.norm()
             << endl;
        cout << "1^T * finite difference grad = "
             << ones.dot(finiteDifferenceGradient) /
                    finiteDifferenceGradient.norm()
             << endl;
    }

    return worstErr;
}

Vector<double> ConePlacer::computePhi(const Vector<double>& u) {
    Vector<double> rhs  = Mii * u;
    Vector<double> ones = Vector<double>::Constant(u.rows(), 1);

    if (abs((ones.dot(rhs))) / rhs.norm() > 1e-8) {
        cerr << "phi solve rhs has nonzero mean. Relative err "
             << abs(ones.dot(rhs)) / rhs.norm() << endl;
    }

    // TODO: it should be Lii.transpose(), but Lii is symmetric, so this
    // should be fine
    if (Lsolver == nullptr) {
        Lsolver = std::unique_ptr<PositiveDefiniteSolver<double>>(
            new PositiveDefiniteSolver<double>(Lii));
    }

    Vector<double> phi      = Lsolver->solve(rhs);
    Vector<double> residual = Lii * phi - rhs;
    if (residual.norm() / phi.norm() > 1e-8) {
        cerr << "phi solve failed with err " << residual.norm()
             << "   and relative error " << residual.norm() / phi.norm()
             << endl;
    }

    return phi;
    // return projectOutConstant(phi);
}

Vector<double> ConePlacer::computeU(const Vector<double>& mu) {
    Vector<double> rhs = Omegaii - mu;
    if (Lsolver == nullptr) {
        Lsolver = std::unique_ptr<PositiveDefiniteSolver<double>>(
            new PositiveDefiniteSolver<double>(Lii));
    }

    Vector<double> u = Lsolver->solve(rhs);
    // u                = projectOutConstant(u);

    Vector<double> residual = Lii * u - rhs;
    if (residual.norm() / rhs.norm() > 1e-4) {
        cerr << "u solve failed with relative err "
             << residual.norm() / rhs.norm() << endl;
    }
    return u;
    // return projectOutConstant(u);
}

Vector<double> ConePlacer::projectOutConstant(const Vector<double>& vec) {
    Vector<double> ones = Vector<double>::Constant(vec.rows(), 1);
    double s            = ones.dot(Mii * vec) / ones.dot(Mii * ones);

    Vector<double> newVec = vec - s * ones;

    double err = ones.dot(Mii * newVec);
    if (abs(err) / newVec.norm() > 1e-4) {
        cerr << "projectOutConstant failed? err " << err << " and relative err "
             << err / newVec.norm() << ". s = " << s
             << "\t old constant part: " << ones.dot(Mii * vec)
             << "\t total surface area: " << ones.dot(Mii * ones) << endl;
    }

    return vec - s * ones;
}

double ConePlacer::computeDistortionEnergy(const Vector<double>& mu,
                                           double lambda) {
    Vector<double> u = computeU(mu);
    return 0.5 * u.transpose() * Mii * u;
}

double ConePlacer::Lagrangian(const Vector<double>& mu, const Vector<double>& u,
                              const Vector<double>& phi) {
    // L = u^T M u - φ^T L u + φ^T Ω - φ^T μ
    return 0.5 * u.dot(Mii * u) - phi.dot(Lii * u) + phi.dot(Omegaii) -
           phi.dot(mu);
}

Vector<double> stackVectors(const std::vector<Vector<double>>& vs) {
    size_t n = 0;
    for (const Vector<double>& v : vs) n += v.rows();

    Vector<double> stack(n);
    size_t iStack = 0;
    for (const Vector<double>& v : vs) {
        for (size_t iV = 0; iV < (size_t)v.rows(); ++iV) {
            stack(iStack++) = v(iV);
        }
    }
    return stack;
}

std::vector<Vector<double>> unstackVectors(const Vector<double>& bigV,
                                           const std::vector<size_t>& sizes) {
    std::vector<Vector<double>> unstack;
    size_t iStack = 0;
    for (size_t n : sizes) {
        Vector<double> v(n);
        for (size_t iV = 0; iV < n; ++iV) {
            v(iV) = bigV(iStack + iV);
        }
        iStack += n;
        unstack.push_back(v);
    }
    return unstack;
}
