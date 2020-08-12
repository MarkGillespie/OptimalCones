#include "ConePlacer.h"

ConePlacer::ConePlacer(ManifoldSurfaceMesh& mesh_, VertexPositionGeometry& geo_)
    : mesh(mesh_), geo(geo_) {
    vIdx = mesh.getVertexIndices();

    geo.requireCotanLaplacian();
    Lii = geo.cotanLaplacian;

    geo.requireVertexLumpedMassMatrix();
    Mii = geo.vertexLumpedMassMatrix;

    geo.requireVertexGaussianCurvatures();
    Omega = geo.vertexGaussianCurvatures.toVector();
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

std::array<Vector<double>, 2>
ConePlacer::computeRegularizedMeasure(Vector<double> u, Vector<double> phi,
                                      double lambda, double gamma) {
    size_t n             = mesh.nVertices();
    double residualNorm2 = 100;

    size_t iter = 0;
    while (residualNorm2 > 1e-5 && iter++ < 50) {
        Vector<double> b = regularizedResidual(u, phi, lambda, gamma);

        SparseMatrix<double> DF = -computeRegularizedDF(u, phi, lambda, gamma);

        SquareSolver<double> solver(DF);

        std::vector<Vector<double>> step =
            unstackVectors(solver.solve(b), {n, n});

        u += step[0];
        phi += step[1];

        residualNorm2 = step[0].squaredNorm() + step[1].squaredNorm();
        if (verbose)
            cout << "\t\t" << iter << "\t| residual: " << residualNorm2 << endl;
    }

    return {u, phi};
}

std::array<VertexData<double>, 3>
ConePlacer::computeOptimalMeasure(Vector<double> u, Vector<double> phi,
                                  double lambda) {
    Vector<double> mu    = proj(phi, lambda);
    size_t n             = mesh.nVertices();
    double residualNorm2 = 100;

    size_t iter = 0;
    while (residualNorm2 > 1e-12 && iter++ < 1000) {

        Vector<double> b = residual(u, phi, mu, lambda);

        SparseMatrix<double> DF = -computeDF(u, phi, mu, lambda);

        SquareSolver<double> solver(DF);

        std::vector<Vector<double>> step =
            unstackVectors(solver.solve(b), {n, n, n});

        u += step[0];
        phi += step[1];
        mu += step[2];

        residualNorm2 = step[0].squaredNorm() + step[1].squaredNorm() +
                        step[2].squaredNorm();
        if (verbose)
            cout << "\t" << iter << "\t| residual: " << residualNorm2 << endl;
    }

    Vector<double> r = residual(u, phi, mu, lambda);
    if (verbose) cout << "\t final residual: " << r.lpNorm<2>() << endl;

    VertexData<double> uData(mesh, u);
    VertexData<double> phiData(mesh, phi);
    VertexData<double> muData(mesh, mu);

    return {uData, phiData, muData};
}

Vector<double> ConePlacer::regularizedResidual(const Vector<double>& u,
                                               const Vector<double>& phi,
                                               double lambda, double gamma) {
    Vector<double> r0 = Lii * u - Omega + proj(phi, lambda) / gamma;
    Vector<double> r1 = Lii.transpose() * phi - Mii * u;
    return stackVectors({r0, r1});
}

Vector<double> ConePlacer::residual(const Vector<double>& u,
                                    const Vector<double>& phi,
                                    const Vector<double>& mu, double lambda) {
    Vector<double> r0 = Lii * u - Omega + mu;
    Vector<double> r1 = Lii.transpose() * phi - Mii * u;
    Vector<double> r2 = mu - proj(Vector<double>(phi + mu), lambda);
    return stackVectors({r0, r1, r2});
}

SparseMatrix<double> ConePlacer::computeDF(const Vector<double>& u,
                                           const Vector<double>& phi,
                                           const Vector<double>& mu,
                                           double lambda) {
    SparseMatrix<double> zeros(mesh.nVertices(), mesh.nVertices());
    SparseMatrix<double> id(mesh.nVertices(), mesh.nVertices());
    id.setIdentity();
    SparseMatrix<double> Dphimu = D(phi + mu, lambda);
    // clang-format off
    SparseMatrix<double> top = horizontalStack<double>({Lii, zeros, id});
    SparseMatrix<double> mid = horizontalStack<double>({-Mii, Lii.transpose(), zeros});
    SparseMatrix<double> bot = horizontalStack<double>({zeros, -Dphimu, id - Dphimu});
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

Vector<double> ConePlacer::proj(Vector<double> x, double lambda) {
    for (size_t iV = 0; iV < (size_t)x.rows(); ++iV) {
        // x(iV) = fmin(fmax(x(iV), -lambda), lambda);
        x(iV) = fmax(0, x(iV) - lambda) + fmin(0, x(iV) + lambda);
    }
    return x;
}


SparseMatrix<double> ConePlacer::D(const Vector<double>& x, double lambda) {
    std::vector<Eigen::Triplet<double>> T;
    for (size_t iV = 0; iV < (size_t)x.rows(); ++iV) {
        if (x(iV) > lambda || x(iV) < -lambda) {
            T.emplace_back(iV, iV, 1);
        }
        // if (-lambda <= x(iV) && x(iV) < lambda) {
        //     T.emplace_back(iV, iV, 1);
        // }
    }
    size_t n = mesh.nVertices();
    SparseMatrix<double> M(n, n);
    M.setFromTriplets(std::begin(T), std::end(T));

    return M;
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

void ConePlacer::setVerbose(bool verb) { verbose = verb; }
