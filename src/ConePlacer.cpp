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

cholmod_sparse* stolen_toCholmod(SparseMatrix<double>& A,
                                 CholmodContext& context, SType stype) {

    A.makeCompressed();

    // Allocate spase
    size_t Nentries = A.nonZeros();
    size_t Ncols    = A.cols();
    size_t Nrows    = A.rows();

    cholmod_sparse* cMat = cholmod_l_allocate_sparse(
        Nrows, Ncols, Nentries, true, true, 0, CHOLMOD_REAL, context);

    // Pull out useful pointers
    double* values               = (double*)cMat->x;
    SuiteSparse_long* rowIndices = (SuiteSparse_long*)cMat->i;
    SuiteSparse_long* colStart   = (SuiteSparse_long*)cMat->p;

    // Copy
    for (size_t iEntry = 0; iEntry < Nentries; iEntry++) {
        values[iEntry]     = A.valuePtr()[iEntry];
        rowIndices[iEntry] = A.innerIndexPtr()[iEntry];
    }
    for (size_t iCol = 0; iCol < Ncols; iCol++) {
        colStart[iCol] = A.outerIndexPtr()[iCol];
    }
    colStart[Ncols] = Nentries;

    return cMat;
}

void stolen_umfSolve(size_t N, cholmod_sparse* mat, Vector<double>& x,
                     const Vector<double>& rhs) {
    SuiteSparse_long* cMat_p = (SuiteSparse_long*)mat->p;
    SuiteSparse_long* cMat_i = (SuiteSparse_long*)mat->i;
    double* cMat_x           = (double*)mat->x;

    void* symbolicFac = nullptr;
    void* numericFac  = nullptr;
    umfpack_dl_symbolic(N, N, cMat_p, cMat_i, cMat_x, &symbolicFac, NULL, NULL);
    umfpack_dl_numeric(cMat_p, cMat_i, cMat_x, symbolicFac, &numericFac, NULL,
                       NULL);

    x = Vector<double>(N);
    umfpack_dl_solve(UMFPACK_A, cMat_p, cMat_i, cMat_x, &(x[0]), &(rhs[0]),
                     numericFac, NULL, NULL);
}

std::array<Vector<double>, 2>
ConePlacer::computeRegularizedMeasure(Vector<double> u, Vector<double> phi,
                                      double lambda, double gamma) {
    size_t n             = mesh.nVertices();
    double residualNorm2 = 100;

    /*
    CholmodContext context;
    cholmod_sparse* cMat = nullptr;
    void* symbolicFac    = nullptr;
    void* numericFac     = nullptr;

    SparseMatrix<double> DF = computeRegularizedDF(u, phi, lambda, gamma);
    cMat                    = stolen_toCholmod(DF, context, SType::UNSYMMETRIC);

    DF.makeCompressed();
    size_t nRows = DF.rows();
    size_t nCols = DF.cols();
    size_t nnz   = DF.nonZeros();

    double* values               = (double*)cMat->x;
    SuiteSparse_long* rowIndices = (SuiteSparse_long*)cMat->i;
    SuiteSparse_long* colStart   = (SuiteSparse_long*)cMat->p;

    // Record which entries are in the upper right corner
    std::vector<size_t> dPositions;
    dPositions.reserve(n);
    for (size_t iCol = n; iCol < nCols; ++iCol) {
        size_t cStart = colStart[iCol];
        size_t cEnd   = colStart[iCol + 1];
        for (size_t iEntry = cStart; iEntry < cEnd; ++iEntry) {
            if (rowIndices[iEntry] < (SuiteSparse_long)n) {
                dPositions.push_back(iEntry);
            }
        }
    }

    SuiteSparse_long N       = nRows;
    SuiteSparse_long* cMat_p = (SuiteSparse_long*)cMat->p;
    SuiteSparse_long* cMat_i = (SuiteSparse_long*)cMat->i;
    double* cMat_x           = (double*)cMat->x;
    umfpack_dl_symbolic(N, N, cMat_p, cMat_i, cMat_x, &symbolicFac, NULL, NULL);
    */

    size_t iter = 0;
    while (residualNorm2 > 1e-5 && iter++ < 50) {
        Vector<double> b = -regularizedResidual(u, phi, lambda, gamma);

        /*
        std::vector<double> newDiag = Dvec(phi, lambda);

        for (size_t iV = 0; iV < n; ++iV)
            cMat_x[dPositions[iV]] = newDiag[iV] / gamma;

        Vector<double> stackedStep(nRows);

        umfpack_dl_numeric(cMat_p, cMat_i, cMat_x, symbolicFac, &numericFac,
                           NULL, NULL);

        umfpack_dl_solve(UMFPACK_A, cMat_p, cMat_i, cMat_x, &(stackedStep[0]),
                         &(b[0]), numericFac, NULL, NULL);
        */

        SparseMatrix<double> DF = computeRegularizedDF(u, phi, lambda, gamma);
        Vector<double> stackedStep = solveSquare(DF, b);

        std::vector<Vector<double>> step = unstackVectors(stackedStep, {n, n});

        u += step[0];
        phi += step[1];

        residualNorm2 = step[0].squaredNorm() + step[1].squaredNorm();
        if (verbose)
            cout << "\t\t" << iter << "\t| residual: " << residualNorm2 << endl;
    }

    // Free Cholmod stuff
    // if (cMat != nullptr) cholmod_l_free_sparse(&cMat, context);
    // if (symbolicFac != nullptr) umfpack_dl_free_symbolic(&symbolicFac);
    // if (numericFac != nullptr) umfpack_dl_free_numeric(&numericFac);

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

        SparseMatrix<double> DF = computeDF(u, phi, mu, lambda);

        SquareSolver<double> solver(DF);

        std::vector<Vector<double>> step =
            unstackVectors(solver.solve(-b), {n, n, n});

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

    std::vector<double> diag = Dvec(x, lambda);
    size_t n                 = mesh.nVertices();

    for (size_t iE = 0; iE < n; ++iE) T.emplace_back(iE, iE, diag[iE]);

    SparseMatrix<double> M(n, n);
    M.setFromTriplets(std::begin(T), std::end(T));

    return M;
}

std::vector<double> ConePlacer::Dvec(const Vector<double>& x, double lambda) {
    std::vector<double> result;
    for (size_t iV = 0; iV < (size_t)x.rows(); ++iV) {
        if (x(iV) > lambda || x(iV) < -lambda) {
            result.push_back(1);
        } else {
            // Push 0s to maintain sparsity pattern
            result.push_back(0);
        }
        // if (-lambda <= x(iV) && x(iV) < lambda) {
        //     result.push_back(1);
        // }
    }
    return result;
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
