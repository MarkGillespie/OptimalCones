#include "ConePlacer.h"

ConePlacer::ConePlacer(ManifoldSurfaceMesh& mesh_, VertexPositionGeometry& geo_,
                       const VertexData<double>& shift_)
    : mesh(mesh_), geo(geo_) {
    vIdx = mesh.getVertexIndices();

    isInterior = Vector<bool>(mesh.nVertices());
    for (Vertex v : mesh.vertices()) {
        isInterior(vIdx[v]) = !v.isBoundary();
    }
    nVertices = mesh.nVertices();
    nInterior = mesh.nInteriorVertices();
    nBoundary = nVertices - nInterior;

    geo.requireCotanLaplacian();
    SparseMatrix<double> L = geo.cotanLaplacian;
    // if (nBoundary == 0)
    L += 1e-12 * speye(nVertices);

    BlockDecompositionResult<double> Ldecomp =
        blockDecomposeSquare(L, isInterior);

    Lii = Ldecomp.AA;
    Lib = Ldecomp.AB;

    geo.requireVertexGalerkinMassMatrix();
    SparseMatrix<double> M = geo.vertexGalerkinMassMatrix;
    BlockDecompositionResult<double> Mdecomp =
        blockDecomposeSquare(M, isInterior);
    Mii = Mdecomp.AA;

    geo.requireVertexGaussianCurvatures();
    VertexData<double> Omega = geo.vertexGaussianCurvatures;
    Omegaii                  = Vector<double>(mesh.nInteriorVertices());
    shiftii                  = Vector<double>(mesh.nInteriorVertices());

    std::vector<Eigen::Triplet<double>> ET, RT, WeT;

    size_t iV              = 0;
    bool complainedAlready = false;
    for (Vertex v : mesh.vertices()) {
        if (!v.isBoundary()) {
            shiftii(iV)   = shift_[v];
            Omegaii(iV++) = Omega[v];
            if (abs(Omega[v]) > 1e-8 && !complainedAlready) {
                std::cerr << "Error: non-flat interior vertex" << std::endl;
                complainedAlready = true;
                // exit(1);
            }
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
    Vector<double> muInterior = getInterior(mu.toVector());

    if (Liisolver == nullptr) {
        Liisolver = std::unique_ptr<PositiveDefiniteSolver<double>>(
            new PositiveDefiniteSolver<double>(Lii));
    }

    Vector<double> uInterior = Liisolver->solve(Omegaii - muInterior);

    return VertexData<double>(mesh, extendInteriorByZero(uInterior));
}

VertexData<double> ConePlacer::computePhi(const VertexData<double>& u) {
    Vector<double> uInterior = getInterior(u.toVector());

    if (Liisolver == nullptr) {
        Liisolver = std::unique_ptr<PositiveDefiniteSolver<double>>(
            new PositiveDefiniteSolver<double>(Lii));
    }

    Vector<double> phiInterior = Liisolver->solve(Mii * uInterior);

    return VertexData<double>(mesh, extendInteriorByZero(phiInterior));
}

double ConePlacer::projErr(const VertexData<double>& mu,
                           const VertexData<double>& phi) {
    Vector<double> muVec  = getInterior(mu.toVector());
    Vector<double> phiVec = getInterior(phi.toVector());
    return 0;
}

double ConePlacer::L2Energy(const VertexData<double>& u) {
    Vector<double> uInterior = getInterior(u.toVector());

    return 0.5 * uInterior.dot(Mii * uInterior);
}

void ConePlacer::setVerbose(bool verb) { verbose = verb; }

std::array<Vector<double>, 2>
ConePlacer::computeRegularizedMeasure(Vector<double> u, Vector<double> phi,
                                      double lambda, double gamma) {
    size_t n = mesh.nInteriorVertices();

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
                 << "\t| u norm: " << sqrt(u.dot(Mii * u))
                 << "\t| phi norm: " << sqrt(phi.dot(Mii * phi)) << endl;
    }

    return {u, phi};
}

std::array<VertexData<double>, 3>
ConePlacer::computeOptimalMeasure(Vector<double> u, Vector<double> phi,
                                  double lambda) {
    Vector<double> mu = P(phi, lambda);
    size_t n          = mesh.nInteriorVertices();

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
                 << "\t| u norm: " << sqrt(u.dot(Mii * u))
                 << "\t| phi norm: " << sqrt(phi.dot(Mii * phi))
                 << "\t mu norm: " << mu.lpNorm<1>() << endl;
    }

    Vector<double> r = residual(u, phi, mu, lambda);
    if (verbose) {
        cout << "\t final residual: " << r.lpNorm<2>() << endl;

        Vector<double> ones = Vector<double>::Constant(u.rows(), 1);
        cout << "\t 1^T M u: " << ones.dot(Mii * u) << endl;
        cout << "\t 1^T M phi: " << ones.dot(Mii * phi) << endl;
        cout << "\t 1^T (Omega - mu): " << ones.dot(Omegaii - mu) << endl;
    }

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
        double netConeAngle = muData.toVector().sum();
        cerr << "net cone angle: " << netConeAngle << endl;
        cerr << "total (interior) cone angle: " << mu.lpNorm<1>()
             << "\t L2 energy: " << 0.5 * u.dot(Mii * u) << endl;
    }

    return {uData, phiData, muData};
}

Vector<double> ConePlacer::residual(const Vector<double>& u,
                                    const Vector<double>& phi,
                                    const Vector<double>& mu, double lambda) {
    Vector<double> r0 = Lii * (u - shiftii) - Omegaii + mu;
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
    SparseMatrix<double> Dphimu = D(phi + mu, lambda);
    // clang-format off
    SparseMatrix<double> top = horizontalStack<double>({Lii,   zeros,           id});
    SparseMatrix<double> mid = horizontalStack<double>({- Mii, Lii.transpose(), zeros});
    SparseMatrix<double> bot = horizontalStack<double>({zeros, -Dphimu,         id-Dphimu});
    // clang-format on

    return verticalStack<double>({top, mid, bot});
}

Vector<double> ConePlacer::regularizedResidual(const Vector<double>& u,
                                               const Vector<double>& phi,
                                               double lambda, double gamma) {
    Vector<double> r0 = Lii * (u - shiftii) - Omegaii + P(phi, lambda) / gamma;
    Vector<double> r1 = Lii.transpose() * phi - Mii * u;
    return stackVectors({r0, r1});
}

SparseMatrix<double> ConePlacer::computeRegularizedDF(const Vector<double>& u,
                                                      const Vector<double>& phi,
                                                      double lambda,
                                                      double gamma) {
    SparseMatrix<double> id(mesh.nInteriorVertices(), mesh.nInteriorVertices());
    id.setIdentity();
    // SparseMatrix<double> topRight = -(id - D(phi, lambda)) / gamma;
    SparseMatrix<double> topRight = D(phi, lambda) / gamma;

    SparseMatrix<double> top = horizontalStack<double>({Lii, topRight});
    SparseMatrix<double> bot = horizontalStack<double>({-Mii, Lii.transpose()});

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
    Vector<double> rhs  = Mii * u;
    Vector<double> ones = Vector<double>::Constant(u.rows(), 1);

    if (abs((ones.dot(rhs))) / rhs.norm() > 1e-8) {
        cerr << "phi solve rhs has nonzero mean. Relative err "
             << abs(ones.dot(rhs)) / rhs.norm() << endl;
    }

    // TODO: it should be Lii.transpose(), but Lii is symmetric, so this
    // should be fine
    if (Liisolver == nullptr) {
        Liisolver = std::unique_ptr<PositiveDefiniteSolver<double>>(
            new PositiveDefiniteSolver<double>(Lii));
    }

    Vector<double> phi      = Liisolver->solve(rhs);
    Vector<double> residual = Lii * phi - rhs;
    if (residual.norm() / phi.norm() > 1e-8) {
        cerr << "phi solve failed with err " << residual.norm()
             << "   and relative error " << residual.norm() / phi.norm()
             << endl;
    }

    return phi;
}

Vector<double> ConePlacer::computeU(const Vector<double>& mu) {
    Vector<double> rhs = Omegaii - mu;
    if (Liisolver == nullptr) {
        Liisolver = std::unique_ptr<PositiveDefiniteSolver<double>>(
            new PositiveDefiniteSolver<double>(Lii));
    }

    Vector<double> u = Liisolver->solve(rhs);

    Vector<double> residual = Lii * u - rhs;
    if (residual.norm() / rhs.norm() > 1e-4) {
        cerr << "u solve failed with relative err "
             << residual.norm() / rhs.norm() << endl;
    }
    return u;
}

double ConePlacer::computeDistortionEnergy(const Vector<double>& mu,
                                           double lambda) {
    Vector<double> u = computeU(mu);
    return 0.5 * u.transpose() * Mii * u;
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

Vector<double> ConePlacer::getInterior(const Vector<double>& vec) {
    return splitInteriorBoundary(vec)[0];
}

Vector<double> ConePlacer::getBoundary(const Vector<double>& vec) {
    return splitInteriorBoundary(vec)[1];
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

GreedyPlacer::GreedyPlacer(ManifoldSurfaceMesh& mesh_,
                           VertexPositionGeometry& geo_)
    : mesh(mesh_), geo(geo_) {

    VertexData<size_t> vIdx = mesh.getVertexIndices();
    Vector<bool> isInterior(mesh.nVertices());
    for (Vertex v : mesh.vertices()) {
        isInterior(vIdx[v]) = !v.isBoundary();
    }

    nVertices = mesh.nVertices();
    nInterior = mesh.nInteriorVertices();
    nBoundary = nVertices - nInterior;

    geo.requireCotanLaplacian();
    SparseMatrix<double> L = geo.cotanLaplacian;
    L += 1e-12 * speye(nVertices);

    BlockDecompositionResult<double> Ldecomp =
        blockDecomposeSquare(L, isInterior);

    Lii = Ldecomp.AA;

    geo.requireVertexGalerkinMassMatrix();
    SparseMatrix<double> M = geo.vertexGalerkinMassMatrix;
    BlockDecompositionResult<double> Mdecomp =
        blockDecomposeSquare(M, isInterior);
    Mii = Mdecomp.AA;

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

std::vector<double>
GreedyPlacer::computeTargetAngles(std::vector<Vertex> cones) {

    std::vector<double> coneAngles;

    if (cones.empty()) {
        return coneAngles;
    }

    size_t nCones = cones.size();

    VertexData<size_t> interiorIndex(mesh, 0);
    size_t iV = 0;
    for (Vertex v : mesh.vertices()) {
        if (!v.isBoundary()) {
            interiorIndex[v] = iV++;
        }
    }

    // initialize cone angles
    for (Vertex v : cones) coneAngles.push_back(Omegaii(interiorIndex[v]));

    // extract submatrices
    if (nCones < nInterior) {


        Vector<bool> isInteriorFlat = Vector<bool>::Constant(nInterior, true);
        for (Vertex v : cones) isInteriorFlat(interiorIndex[v]) = false;

        Vector<size_t> coneIndex(nInterior);
        Vector<size_t> flatIndex(nInterior);
        Vector<double> omegaFlat(nInterior - nCones);

        size_t iC = 0;
        size_t iF = 0;
        for (size_t iV = 0; iV < nInterior; ++iV) {
            if (isInteriorFlat(iV)) {
                omegaFlat(iF) = Omegaii(iV);
                flatIndex(iV) = iF++;
            } else {
                coneIndex(iV) = iC++;
            }
        }

        BlockDecompositionResult<double> Liidecomp =
            blockDecomposeSquare(Lii, isInteriorFlat);

        SparseMatrix<double> Lff = Liidecomp.AA;
        SparseMatrix<double> Lfc = Liidecomp.AB;


        // compute target curvatures
        for (size_t iC = 0; iC < nCones; ++iC) {
            Vector<double> delta = Vector<double>::Zero(nCones);
            delta(iC)            = 1;
            Vector<double> rhs   = -Lfc * delta;

            Vector<double> Gn = solvePositiveDefinite(Lff, rhs);

            // Cs = Ks + Gn^T Kn
            coneAngles[iC] += Gn.dot(omegaFlat);
        }
    }

    return coneAngles;
}

VertexData<double>
GreedyPlacer::computeInitialScaleFactors(VertexData<double> vertexCurvatures) {
    geo.requireCotanLaplacian();

    double totalCurvature = 2 * M_PI * mesh.eulerCharacteristic();
    double avgCurvature   = totalCurvature / (double)mesh.nVertices();

    Vector<double> weightedCurvature = -vertexCurvatures.toVector();
    Vector<double> constantCurvature =
        Vector<double>::Constant(mesh.nVertices(), avgCurvature);

    Vector<double> curvatureDifference = weightedCurvature - constantCurvature;

    Vector<double> result =
        solvePositiveDefinite(geo.cotanLaplacian, curvatureDifference);

    vertexCurvatures.fromVector(result);
    return vertexCurvatures;
}

VertexData<double>
GreedyPlacer::computeScaleFactors(VertexData<double> vertexCurvatures,
                                  const std::vector<Vertex> cones, int vSkip) {
    geo.requireCotanLaplacian();

    // interior vertices are the ones that aren't cones
    VertexData<size_t> vIdx = mesh.getVertexIndices();
    Vector<bool> isInterior = Vector<bool>::Constant(mesh.nVertices(), true);

    size_t nCones = 0;
    for (size_t iV = 0; iV < cones.size(); ++iV) {
        if ((int)iV != vSkip) {
            isInterior[vIdx[cones[iV]]] = false;
            nCones++;
        }
    }
    // If the mesh has boundary, make every boundary vertex a cone
    for (BoundaryLoop b : mesh.boundaryLoops()) {
        for (Vertex v : b.adjacentVertices()) {
            isInterior[vIdx[v]] = false;
            nCones++;
        }
    }

    BlockDecompositionResult<double> laplacianBlocks =
        blockDecomposeSquare(geo.cotanLaplacian, isInterior, true);

    Vector<double> weightedCurvature = -vertexCurvatures.toVector();

    Vector<double> coneCurvature, interiorCurvature;
    decomposeVector(laplacianBlocks, weightedCurvature, interiorCurvature,
                    coneCurvature);

    Vector<double> interiorResult =
        solvePositiveDefinite(laplacianBlocks.AA, interiorCurvature);
    Vector<double> zero = Eigen::VectorXd::Zero(nCones);
    Vector<double> totalResult =
        reassembleVector(laplacianBlocks, interiorResult, zero);

    vertexCurvatures.fromVector(totalResult);
    return vertexCurvatures;
}

VertexData<double> GreedyPlacer::niceCones(size_t nCones,
                                           size_t gaussSeidelIterations) {
    // Gaussian curvature at interior vertices, geodesic curvature at boundary
    // vertices
    VertexData<double> curvature(mesh);
    geo.requireVertexAngleSums();
    for (Vertex v : mesh.vertices()) {
        if (v.isBoundary()) {
            curvature[v] = PI - geo.vertexAngleSums[v];
        } else {
            curvature[v] = 2 * PI - geo.vertexAngleSums[v];
        }
    }

    // Place first cone by approximately uniformizing to a constant curvature
    // metric and picking the vertex with worst scale factor (We would flatten,
    // except you can't necessarily flatten with one cone)
    VertexData<double> scaleFactors = computeInitialScaleFactors(curvature);
    size_t worstVertex =
        argmax(scaleFactors, [](double x) { return std::abs(x); });
    std::vector<Vertex> cones{mesh.vertex(worstVertex)};

    // Place remaining cones by approximately flattening and picking the vertex
    // with worst scale factor
    for (size_t iCone = 1; iCone < nCones; ++iCone) {
        VertexData<double> scaleFactors = computeScaleFactors(curvature, cones);
        size_t worstVertex =
            argmax(scaleFactors, [](double x) { return std::abs(x); });
        cones.push_back(mesh.vertex(worstVertex));
    }

    // Place remaining cones by approximately flattening and picking the vertex
    // with worst scale factor
    for (size_t iGS = 0; iGS < gaussSeidelIterations; ++iGS) {
        for (size_t iCone = 0; iCone < nCones; ++iCone) {
            VertexData<double> scaleFactors =
                computeScaleFactors(curvature, cones, iCone);
            size_t worstVertex =
                argmax(scaleFactors, [](double x) { return std::abs(x); });
            cones[iCone] = mesh.vertex(worstVertex);
        }
    }

    std::vector<double> coneAngles = computeTargetAngles(cones);

    VertexData<double> mu(mesh, 0);
    for (size_t iC = 0; iC < cones.size(); ++iC) {
        mu[cones[iC]] = coneAngles[iC];
    }

    return mu;
}
