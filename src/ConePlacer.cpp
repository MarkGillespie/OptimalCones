#include "ConePlacer.h"

ConePlacer::ConePlacer(ManifoldSurfaceMesh& mesh_, VertexPositionGeometry& geo_)
    : mesh(mesh_), geo(geo_) {
    vIdx = mesh.getVertexIndices();

    nVertices = mesh.nVertices();

    geo.requireCotanLaplacian();
    L = geo.cotanLaplacian;
    // if (nBoundary == 0)
    L += 1e-12 * speye(nVertices);

    geo.requireVertexLumpedMassMatrix();
    M = geo.vertexLumpedMassMatrix;

    geo.requireVertexAngleSums();
    Omega = Vector<double>::Constant(nVertices, 0);

    for (Vertex v : mesh.vertices()) {
        if (v.isBoundary()) {
            Omega(vIdx[v]) = M_PI - geo.vertexAngleSums[v];
        } else {
            Omega(vIdx[v]) = 2 * M_PI - geo.vertexAngleSums[v];
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

    Vector<double> u   = Vector<double>::Constant(mesh.nVertices(), 0);
    Vector<double> phi = Vector<double>::Constant(mesh.nVertices(), 0);
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

VertexData<double> ConePlacer::contractClusters(VertexData<double>& mu,
                                                bool fancy) {
    if (fancy) {
        // Diffuse mu slightly to smooth it out
        geo.requireMeshLengthScale();
        double dt = 1e-12;
        // double dt = 1e-8;
        // double dt = 1e-6 * geo.meshLengthScale;

        geo.requireCotanLaplacian();
        geo.requireVertexGalerkinMassMatrix();
        const SparseMatrix<double>& L = geo.cotanLaplacian;
        const SparseMatrix<double>& M = geo.vertexGalerkinMassMatrix;

        geo.requireVertexDualAreas();
        Vector<double> MinvMu = mu.toVector();
        for (Vertex v : mesh.vertices())
            MinvMu(vIdx[v]) /= geo.vertexDualAreas[v];

        SparseMatrix<double> diffuse = M + dt * L;

        Vector<double> interiorMu = mu.toVector();
        for (Vertex v : mesh.vertices()) {
            if (v.isBoundary()) interiorMu(vIdx[v]) = 0;
        }

        Vector<double> fuzzyMu = M * solvePositiveDefinite(diffuse, interiorMu);
        for (Vertex v : mesh.vertices()) {
            if (v.isBoundary()) fuzzyMu(vIdx[v]) = mu[v];
        }

        mu = VertexData<double>(mesh, fuzzyMu);

        polyscope::getSurfaceMesh("mesh")->addVertexScalarQuantity("fuzzy mu",
                                                                   mu);
    }

    double tol = 1e-8;
    VertexData<bool> visited(mesh, false);
    VertexData<double> contracted(mesh, 0);

    auto contractCluster = [&](Vertex v) {
        if (visited[v] || abs(mu[v]) < tol) {
            visited[v] = true;
            return;
        } else if (v.isBoundary()) {
            contracted[v] = mu[v];
        }

        std::vector<Vertex> cluster;
        std::deque<Vertex> toVisit;
        toVisit.push_back(v);
        visited[v] = true;

        while (!toVisit.empty()) {
            Vertex v = toVisit.front();
            toVisit.pop_front();
            cluster.push_back(v);

            if (abs(mu[v]) >= tol) {
                for (Vertex w : v.adjacentVertices()) {
                    if (!visited[w] && !w.isBoundary()) {
                        toVisit.push_back(w);
                        visited[w] = true;
                    }
                }
            }
        }

        std::vector<Vertex> contractedVertices;
        std::vector<double> contractedWeights;

        std::tie(contractedVertices, contractedWeights) =
            approximateCluster(mu, cluster);

        for (size_t iV = 0; iV < contractedVertices.size(); ++iV)
            contracted[contractedVertices[iV]] = contractedWeights[iV];
    };

    for (Vertex v : mesh.vertices()) {
        contractCluster(v);
    }

    return contracted;
}

VertexData<double> ConePlacer::computeU(const VertexData<double>& mu) {
    if (Lsolver == nullptr) {
        Lsolver = std::unique_ptr<PositiveDefiniteSolver<double>>(
            new PositiveDefiniteSolver<double>(L));
    }

    Vector<double> u = Lsolver->solve(Omega - mu.toVector());

    return VertexData<double>(mesh, u);
}

VertexData<double> ConePlacer::computePhi(const VertexData<double>& u) {

    if (Lsolver == nullptr) {
        Lsolver = std::unique_ptr<PositiveDefiniteSolver<double>>(
            new PositiveDefiniteSolver<double>(L));
    }

    Vector<double> phi = Lsolver->solve(M * u.toVector());

    return VertexData<double>(mesh, phi);
}

double ConePlacer::projErr(const VertexData<double>& mu,
                           const VertexData<double>& phi) {
    Vector<double> muVec  = mu.toVector();
    Vector<double> phiVec = phi.toVector();
    return 0;
}

double ConePlacer::L2Energy(const VertexData<double>& u) {
    Vector<double> uVec = u.toVector();

    return 0.5 * uVec.dot(M * uVec);
}

void ConePlacer::setVerbose(bool verb) { verbose = verb; }

std::array<Vector<double>, 2>
ConePlacer::computeRegularizedMeasure(Vector<double> u, Vector<double> phi,
                                      double lambda, double gamma) {
    size_t n = mesh.nVertices();

    Vector<double> b = -regularizedResidual(u, phi, lambda, gamma);

    size_t iter = 0;
    while (b.norm() > 1e-5 && iter++ < 50) {
        SparseMatrix<double> DF = computeRegularizedDF(u, phi, lambda, gamma);
        Vector<double> stackedStep       = solveSquare(DF, b);
        std::vector<Vector<double>> step = unstackVectors(stackedStep, {n, n});

        u += step[0];
        phi += step[1];

        b = -regularizedResidual(u, phi, lambda, gamma);

        if (verbose)
            cout << "\t\t" << iter << "\t| residual: " << b.norm()
                 << "\t| u norm: " << sqrt(u.dot(M * u))
                 << "\t| phi norm: " << sqrt(phi.dot(M * phi)) << endl;
    }

    return {u, phi};
}

std::array<VertexData<double>, 3>
ConePlacer::computeOptimalMeasure(Vector<double> u, Vector<double> phi,
                                  double lambda) {
    Vector<double> mu = P(phi, lambda);
    size_t n          = mesh.nVertices();

    Vector<double> b = -residual(u, phi, mu, lambda);

    if (verbose)
        cout << "\t initial residual: "
             << residual(u, phi, mu, lambda).lpNorm<2>()
             << "\t mu norm: " << mu.lpNorm<1>() << endl;

    size_t iter = 0;
    while (b.norm() > 1e-12 && iter++ < 1000) {
        SparseMatrix<double> DF    = computeDF(u, phi, mu, lambda);
        Vector<double> stackedStep = solveSquare(DF, b);
        std::vector<Vector<double>> step =
            unstackVectors(stackedStep, {n, n, n});

        u += step[0];
        phi += step[1];
        mu += step[2];

        b = -residual(u, phi, mu, lambda);

        if (verbose)
            cout << "\t\t" << iter << "\t| residual: " << b.norm()
                 << "\t| u norm: " << sqrt(u.dot(M * u))
                 << "\t| phi norm: " << sqrt(phi.dot(M * phi))
                 << "\t mu norm: " << mu.lpNorm<1>() << endl;
    }

    Vector<double> r = residual(u, phi, mu, lambda);
    if (verbose) {
        cout << "\t final residual: " << r.lpNorm<2>() << endl;

        Vector<double> ones = Vector<double>::Constant(u.rows(), 1);
        cout << "\t 1^T M u: " << ones.dot(M * u) << endl;
        cout << "\t 1^T M phi: " << ones.dot(M * phi) << endl;
        cout << "\t 1^T (Omega - mu): " << ones.dot(Omega - mu) << endl;
    }

    auto assignVertices = [&](const Vector<double>& vec) {
        VertexData<double> data(mesh, 0);
        size_t iV = 0;
        for (Vertex v : mesh.vertices()) {
            data[v] = vec(iV++);
        }
        return data;
    };

    VertexData<double> uData   = assignVertices(u);
    VertexData<double> phiData = assignVertices(phi);

    Vector<double> fullU   = uData.toVector();
    SparseMatrix<double> L = geo.cotanLaplacian;

    VertexData<double> muData(mesh, Omega - L * fullU);

    if (verbose) {
        double netConeAngle = muData.toVector().sum();
        cerr << "net cone angle: " << netConeAngle << endl;
        cerr << "total (interior) cone angle: " << mu.lpNorm<1>()
             << "\t L2 energy: " << 0.5 * u.dot(M * u) << endl;
    }

    return {uData, phiData, muData};
}

Vector<double> ConePlacer::regularizedResidual(const Vector<double>& u,
                                               const Vector<double>& phi,
                                               double lambda, double gamma) {
    Vector<double> r0 = L * u - Omega + P(phi, lambda) / gamma;
    Vector<double> r1 = L.transpose() * phi - M * u;
    return stackVectors({r0, r1});
}

Vector<double> ConePlacer::residual(const Vector<double>& u,
                                    const Vector<double>& phi,
                                    const Vector<double>& mu, double lambda) {
    Vector<double> r0 = L * u - Omega + mu;
    Vector<double> r1 = L.transpose() * phi - M * u;
    Vector<double> r2 = mu - P(Vector<double>(phi + mu), lambda);
    return stackVectors({r0, r1, r2});
}

SparseMatrix<double> ConePlacer::computeDF(const Vector<double>& u,
                                           const Vector<double>& phi,
                                           const Vector<double>& mu,
                                           double lambda) {
    SparseMatrix<double> zeros(mesh.nVertices(), mesh.nVertices());
    SparseMatrix<double> id(mesh.nVertices(), mesh.nVertices());
    id.setIdentity();
    SparseMatrix<double> Dphi   = D(phi, lambda);
    SparseMatrix<double> Dphimu = D(phi + mu, lambda);
    // clang-format off
    SparseMatrix<double> top = horizontalStack<double>({L,   zeros,           id});
    SparseMatrix<double> mid = horizontalStack<double>({- M, L.transpose(), zeros});
    SparseMatrix<double> bot = horizontalStack<double>({zeros, -Dphimu,         id - Dphimu});
    // clang-format on

    return verticalStack<double>({top, mid, bot});
}

SparseMatrix<double> ConePlacer::computeRegularizedDF(const Vector<double>& u,
                                                      const Vector<double>& phi,
                                                      double lambda,
                                                      double gamma) {
    SparseMatrix<double> topRight = D(phi, lambda) / gamma;

    SparseMatrix<double> top = horizontalStack<double>({L, topRight});
    SparseMatrix<double> bot = horizontalStack<double>({-M, L.transpose()});

    return verticalStack<double>({top, bot});
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

Vector<double> ConePlacer::computePhi(const Vector<double>& u) {
    Vector<double> rhs  = M * u;
    Vector<double> ones = Vector<double>::Constant(u.rows(), 1);

    if (abs((ones.dot(rhs))) / rhs.norm() > 1e-8) {
        cerr << "phi solve rhs has nonzero mean. Relative err "
             << abs(ones.dot(rhs)) / rhs.norm() << endl;
    }

    // TODO: it should be L.transpose(), but L is symmetric, so this
    // should be fine
    if (Lsolver == nullptr) {
        Lsolver = std::unique_ptr<PositiveDefiniteSolver<double>>(
            new PositiveDefiniteSolver<double>(L));
    }

    Vector<double> phi      = Lsolver->solve(rhs);
    Vector<double> residual = L * phi - rhs;
    if (residual.norm() / phi.norm() > 1e-8) {
        cerr << "phi solve failed with err " << residual.norm()
             << "   and relative error " << residual.norm() / phi.norm()
             << endl;
    }

    return phi;
}

Vector<double> ConePlacer::computeU(const Vector<double>& mu) {
    Vector<double> rhs = Omega - mu;
    if (Lsolver == nullptr) {
        Lsolver = std::unique_ptr<PositiveDefiniteSolver<double>>(
            new PositiveDefiniteSolver<double>(L));
    }

    Vector<double> u = Lsolver->solve(rhs);

    Vector<double> residual = L * u - rhs;
    if (residual.norm() / rhs.norm() > 1e-4) {
        cerr << "u solve failed with relative err "
             << residual.norm() / rhs.norm() << endl;
    }
    return u;
}

double ConePlacer::computeDistortionEnergy(const Vector<double>& mu,
                                           double lambda) {
    Vector<double> u = computeU(mu);
    return 0.5 * u.transpose() * M * u;
}

std::pair<std::vector<Vertex>, std::vector<double>>
ConePlacer::approximateCluster(const VertexData<double>& mu,
                               const std::vector<Vertex>& cluster) {
    Vertex mode       = cluster[0];
    double clusterSum = 0;

    for (Vertex v : cluster) {
        clusterSum += mu[v];
        if (abs(mu[v]) > abs(mu[mode])) mode = v;
    }

    return {{mode}, {clusterSum}};
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
