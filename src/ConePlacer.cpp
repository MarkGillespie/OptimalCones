#include "ConePlacer.h"

ConePlacer::ConePlacer(ManifoldSurfaceMesh& mesh_, VertexPositionGeometry& geo_)
    : mesh(mesh_), geo(geo_) {
    vIdx      = mesh.getVertexIndices();
    nVertices = mesh.nVertices();
    nInterior = mesh.nInteriorVertices();
    nBoundary = nVertices - nInterior;

    isInterior = Vector<bool>(mesh.nVertices());
    for (Vertex v : mesh.vertices()) {
        isInterior(vIdx[v]) = !v.isBoundary();
    }

    geo.requireCotanLaplacian();
    L = geo.cotanLaplacian;
    BlockDecompositionResult<double> Ldecomp =
        blockDecomposeSquare(L, isInterior);

    Lii = Ldecomp.AA;
    Lib = Ldecomp.AB;

    geo.requireVertexLumpedMassMatrix();
    M = geo.vertexLumpedMassMatrix;
    BlockDecompositionResult<double> Mdecomp =
        blockDecomposeSquare(M, isInterior);
    Mii = Mdecomp.AA;

    std::cout << "L rows: " << L.rows() << "\tLii rows: " << Lii.rows()
              << "\t M rows: " << Mii.rows()
              << "\tnInteriorVertices: " << mesh.nInteriorVertices() << endl;

    geo.requireVertexGaussianCurvatures();
    std::tie(Omegaii, k) =
        splitInteriorBoundary(geo.vertexGaussianCurvatures.toVector());
    OmegaK = combineInteriorBoundary(Omegaii, k);

    std::vector<Eigen::Triplet<double>> IiT, IbT;
    size_t iI = 0;
    size_t iB = 0;
    for (size_t iV = 0; iV < nVertices; ++iV) {
        if (isInterior(iV)) {
            IiT.emplace_back(iV, iI++, 1);
        } else {
            IbT.emplace_back(iV, iB++, 1);
        }
    }
    Ii = SparseMatrix<double>(nVertices, nInterior);
    Ib = SparseMatrix<double>(nVertices, nBoundary);
    Ii.setFromTriplets(std::begin(IiT), std::end(IiT));
    Ib.setFromTriplets(std::begin(IbT), std::end(IbT));
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

    Vector<double> u   = Vector<double>::Constant(nVertices, 0);
    Vector<double> phi = Vector<double>::Constant(nVertices, 0);
    Vector<double> h   = Vector<double>::Constant(nBoundary, 0);
    double gamma       = 100;

    if (verbose) cout << "Beginning Regularized Solves" << endl;

    for (size_t iS = 0; iS < regularizedSteps; ++iS) {
        std::tie(u, phi, h) =
            computeRegularizedMeasure(u, phi, h, lambda, gamma);
        gamma /= 10;
        if (verbose) cout << "\tFinished iteration " << iS << endl;
    }

    if (verbose) cout << "Beginning Exact Solve" << endl;
    return computeOptimalMeasure(u, phi, h, lambda);
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

std::array<Vector<double>, 3>
ConePlacer::computeRegularizedMeasure(Vector<double> u, Vector<double> phi,
                                      Vector<double> h, double lambda,
                                      double gamma) {

    Vector<double> b    = -regularizedResidual(u, phi, h, lambda, gamma);
    double residualNorm = b.norm();

    Vector<double> ones = Vector<double>::Constant(mesh.nInteriorVertices(), 1);

    size_t iter = 0;
    while (residualNorm > 1e-5 && iter++ < 50) {

        SparseMatrix<double> DF =
            computeRegularizedDF(u, phi, h, lambda, gamma);
        // Vector<double> stackedStep = solve(DF, b);

        // cerr << "solve residual: " << (DF * stackedStep - b).norm()
        //      << "\t relative err: " << (DF * stackedStep - b).norm() /
        //      b.norm()
        //      << endl;

        SparseMatrix<double> DFTDF = DF.transpose() * DF;
        Vector<double> DFTb        = DF.transpose() * b;
        Vector<double> stackedStep = solveSquare(DFTDF, DFTb);

        double solveResidual = (DF * stackedStep - b).norm();
        if (solveResidual > 1e-4 * b.norm()) {
            cerr << "better? solve residual: " << solveResidual
                 << "\t relative err: " << solveResidual / b.norm() << endl;
        }

        std::vector<Vector<double>> step =
            unstackVectors(stackedStep, {nVertices, nVertices, nBoundary});

        u += step[0];
        phi += step[1];
        h += step[2];

        b            = -regularizedResidual(u, phi, h, lambda, gamma);
        residualNorm = b.norm() / u.norm();
        if (verbose)
            cout << "\t\t" << iter << "\t| residual: " << residualNorm
                 << "\t| u norm: " << u.lpNorm<1>()
                 << "\t| phi norm: " << phi.lpNorm<1>() << endl;
    }

    return {u, phi, h};
}

std::array<VertexData<double>, 3>
ConePlacer::computeOptimalMeasure(Vector<double> u, Vector<double> phi,
                                  Vector<double> h, double lambda) {

    Vector<double> phii;
    std::tie(phii, std::ignore) = splitInteriorBoundary(phi);
    Vector<double> mu           = P(phii, lambda);
    size_t n                    = mesh.nInteriorVertices();
    double residualNorm2        = residual(u, phi, mu, h, lambda).lpNorm<2>();

    if (verbose)
        cout << "\t initial residual: " << residualNorm2
             << "\t mu norm: " << mu.lpNorm<1>() << endl;

    size_t iter = 0;
    while (residualNorm2 > 1e-12 && iter++ < 1000) {

        Vector<double> b = -residual(u, phi, mu, h, lambda);

        SparseMatrix<double> DF = computeDF(u, phi, mu, h, lambda);

        // SquareSolver<double> solver(DF);
        // Vector<double> stackedStep = solver.solve(b);

        SparseMatrix<double> DFTDF = DF.transpose() * DF;
        Vector<double> DFTb        = DF.transpose() * b;
        Vector<double> stackedStep = solveSquare(DFTDF, DFTb);


        std::vector<Vector<double>> step = unstackVectors(
            stackedStep, {nVertices, nVertices, nInterior, nBoundary});

        u += step[0];
        phi += step[1];
        mu += step[2];
        h += step[3];

        residualNorm2 = stackedStep.squaredNorm();

        if (verbose)
            cout << "\t" << iter << "\t| residual: " << residualNorm2
                 << "\t u norm: " << u.lpNorm<1>()
                 << "\t phi norm: " << phi.lpNorm<1>()
                 << "\t mu norm: " << mu.lpNorm<1>()
                 << "\t step2 norm: " << step[2].lpNorm<1>() << endl;
    }

    Vector<double> r = residual(u, phi, mu, h, lambda);
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
                                               const Vector<double>& h,
                                               double lambda, double gamma) {
    Vector<double> phii;
    std::tie(phii, std::ignore) = splitInteriorBoundary(phi);

    Vector<double> mu     = P(phii, lambda) / gamma;
    Vector<double> muh    = combineInteriorBoundary(mu, h);
    Vector<double> hTotal = extendBoundaryByZero(h);

    Vector<double> r0 = L * u - OmegaK + muh;
    Vector<double> r1 = L.transpose() * phi - M * u + hTotal;
    return stackVectors({r0, r1});
}

SparseMatrix<double> ConePlacer::computeRegularizedDF(const Vector<double>& u,
                                                      const Vector<double>& phi,
                                                      const Vector<double>& h,
                                                      double lambda,
                                                      double gamma) {

    Vector<double> phii;
    std::tie(phii, std::ignore) = splitInteriorBoundary(phi);
    SparseMatrix<double> Dphi   = Dinterior(phii, lambda) / gamma;

    // clang-format off
    SparseMatrix<double> r0, r1;
    //                              u  phi  h
    r0 = horizontalStack<double>({L,  Dphi, Ib});
    r1 = horizontalStack<double>({-M, L,    Ib});
    // clang-format on

    return verticalStack<double>({r0, r1});
}

Vector<double> ConePlacer::residual(const Vector<double>& u,
                                    const Vector<double>& phi,
                                    const Vector<double>& mu,
                                    const Vector<double>& h, double lambda) {

    Vector<double> phii;
    std::tie(phii, std::ignore) = splitInteriorBoundary(phi);

    Vector<double> muh    = combineInteriorBoundary(mu, h);
    Vector<double> hTotal = extendBoundaryByZero(h);

    Vector<double> r0 = L * u - OmegaK + muh;
    Vector<double> r1 = L.transpose() * phi - M * u + hTotal;
    Vector<double> r2 = mu - P(Vector<double>(phii + mu), lambda);
    return stackVectors({r0, r1, r2});
}

SparseMatrix<double> ConePlacer::computeDF(const Vector<double>& u,
                                           const Vector<double>& phi,
                                           const Vector<double>& mu,
                                           const Vector<double>& h,
                                           double lambda) {
    Vector<double> phii;
    std::tie(phii, std::ignore) = splitInteriorBoundary(phi);

    SparseMatrix<double> zerosa(nVertices, nVertices);
    SparseMatrix<double> zerosai(nVertices, nInterior);
    SparseMatrix<double> zerosia(nInterior, nVertices);
    SparseMatrix<double> zerosib(nInterior, nBoundary);
    SparseMatrix<double> id     = speye(nInterior);
    SparseMatrix<double> Dphimu = D(phii + mu, lambda);
    SparseMatrix<double> IiT    = Ii.transpose();

    // clang-format off
    SparseMatrix<double> r0, r1, r2;
    //                            u        phi            mu           h
    r0 = horizontalStack<double>({L,       zerosa,        Ii,          Ib});
    r1 = horizontalStack<double>({-M,      L,             zerosai,     Ib});
    r2 = horizontalStack<double>({zerosia, -Dphimu * IiT, id - Dphimu, zerosib});
    // clang-format on

    return verticalStack<double>({r0, r1, r2});
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

    Vector<double> diag = Dvec(x, lambda);

    for (size_t iE = 0; iE < (size_t)diag.rows(); ++iE) {
        if (abs(diag(iE)) > 1e-12) T.emplace_back(iE, iE, diag(iE));
    }

    SparseMatrix<double> M(diag.size(), diag.size());
    M.setFromTriplets(std::begin(T), std::end(T));

    return M;
}

SparseMatrix<double> ConePlacer::Dinterior(const Vector<double>& x,
                                           double lambda) {
    std::vector<Eigen::Triplet<double>> T;

    Vector<double> diag = extendInteriorByZero(Dvec(x, lambda));

    for (size_t iE = 0; iE < (size_t)diag.rows(); ++iE) {
        if (abs(diag(iE)) > 1e-12) T.emplace_back(iE, iE, diag(iE));
    }

    SparseMatrix<double> M(diag.size(), diag.size());
    M.setFromTriplets(std::begin(T), std::end(T));

    return M;
}

Vector<double> ConePlacer::Dvec(const Vector<double>& x, double lambda) {
    Vector<double> result(x.rows());
    for (size_t iV = 0; iV < (size_t)x.rows(); ++iV) {
        if (abs(x(iV)) > lambda) {
            result(iV) = 1;
        } else {
            result(iV) = 0;
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

std::array<Vector<double>, 2>
ConePlacer::splitInteriorBoundary(const Vector<double>& vec) {
    Vector<double> interior(nInterior);
    Vector<double> boundary(nBoundary);

    size_t iI = 0;
    size_t iB = 0;
    for (size_t iV = 0; iV < nVertices; ++iV) {
        if (isInterior(iV)) {
            interior(iI++) = vec(iV);
        } else {
            boundary(iB++) = vec(iV);
        }
    }
    return {interior, boundary};
}


Vector<double>
ConePlacer::combineInteriorBoundary(const Vector<double>& interior,
                                    const Vector<double>& boundary) {
    Vector<double> combined(nVertices);
    size_t iI = 0;
    size_t iB = 0;

    for (size_t iV = 0; iV < nVertices; ++iV) {
        if (isInterior(iV)) {
            combined(iV) = interior(iI++);
        } else {
            combined(iV) = boundary(iB++);
        }
    }
    return combined;
}

Vector<double>
ConePlacer::extendBoundaryByZero(const Vector<double>& boundary) {
    return combineInteriorBoundary(Vector<double>::Zero(nInterior), boundary);
}

Vector<double>
ConePlacer::extendInteriorByZero(const Vector<double>& interior) {
    return combineInteriorBoundary(interior, Vector<double>::Zero(nBoundary));
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

SparseMatrix<double> speye(size_t n) {
    SparseMatrix<double> eye(n, n);
    eye.setIdentity();
    return eye;
}
