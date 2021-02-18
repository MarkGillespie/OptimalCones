#include "bff.h"

BFF::BFF(ManifoldSurfaceMesh& mesh_, IntrinsicGeometryInterface& geo_)
    : mesh(mesh_), geo(geo_) {
    vIdx = mesh.getVertexIndices();
    iIdx = VertexData<int>(mesh, -1);
    bIdx = VertexData<int>(mesh, -1);

    geo.requireCotanLaplacian();

    L = geo.cotanLaplacian;

    isInterior = Vector<bool>(mesh.nVertices());
    size_t iB = 0, iI = 0;
    for (Vertex v : mesh.vertices()) {
        if (v.isBoundary()) {
            bIdx[v]             = iB++;
            isInterior(vIdx[v]) = false;
        } else {
            iIdx[v]             = iI++;
            isInterior(vIdx[v]) = true;
        }
    }
    nVertices = mesh.nVertices();
    nInterior = mesh.nInteriorVertices();
    nBoundary = nVertices - nInterior;

    geo.requireCotanLaplacian();
    SparseMatrix<double> L = geo.cotanLaplacian;
    shiftDiagonal(L, 1e-12);

    Ldecomp = blockDecomposeSquare(L, isInterior);

    Lii = Ldecomp.AA;
    Lib = Ldecomp.AB;
    Lbb = Ldecomp.BB;

    Liisolver = std::make_unique<PositiveDefiniteSolver<double>>(Lii);

    geo.requireVertexAngleSums();
    Omegai = Vector<double>(nInterior);
    Omegab = Vector<double>(nBoundary);
    for (Vertex v : mesh.vertices()) {
        if (v.isBoundary()) {
            Omegab(bIdx[v]) = M_PI - geo.vertexAngleSums[v];
        } else {
            Omegai(iIdx[v]) = 2 * M_PI - geo.vertexAngleSums[v];
        }
    }
}

VertexData<Vector2> BFF::flattenMAD() {
    VertexData<double> u(mesh, 0);
    return flattenFromU(u);
}

VertexData<Vector2> BFF::flattenFromU(const VertexData<double>& uData) {

    Vector<double> uBdy, ignore;
    decomposeVector(Ldecomp, uData.toVector(), ignore, uBdy);
    Vector<double> kBdy = dirichletToNeumann(uBdy);

    Vector<double> boundaryX, boundaryY;
    std::tie(boundaryX, boundaryY) =
        tuple_cat(computeRoundedBoundary(uBdy, kBdy));

    Vector<double> interiorX = Liisolver->solve(-Lib * boundaryX);
    Vector<double> interiorY = Liisolver->solve(-Lib * boundaryY);

    VertexData<Vector2> parm(mesh);
    for (Vertex v : mesh.vertices()) {
        if (v.isBoundary()) {
            size_t iV = bIdx[v];
            parm[v]   = Vector2{boundaryX(iV), boundaryY(iV)};
        } else {
            size_t iV = iIdx[v];
            parm[v]   = Vector2{interiorX(iV), interiorY(iV)};
        }
    }

    Vector<double> uInt = Liisolver->solve(Omegai);
    u                   = VertexData<double>(mesh, 0);
    for (Vertex v : mesh.vertices()) {
        if (!v.isBoundary()) {
            u[v] = uInt(iIdx[v]);
        }
    }

    return parm;
}

Vector<double> BFF::dirichletToNeumann(const Vector<double> uBdy) {
    return Omegab - (Lib.transpose() * Liisolver->solve(Omegai - Lib * uBdy)) -
           Lbb * uBdy;
}

std::array<Vector<double>, 2>
BFF::computeRoundedBoundary(const Vector<double>& uBdy,
                            const Vector<double>& kBdy) {

    auto src = [](Edge e) { return e.halfedge().vertex(); };
    auto dst = [](Edge e) { return e.halfedge().next().vertex(); };

    geo.requireEdgeLengths();

    double phi = 0;

    std::vector<Eigen::Triplet<double>> Ntriplets;
    Vector<double> targetLength(nBoundary);
    DenseMatrix<double> T(2, nBoundary);

    for (BoundaryLoop b : mesh.boundaryLoops()) {
        for (Edge e : b.adjacentEdges()) {
            int iV = bIdx[src(e)];
            if (iV < 0 || iV >= (int)nBoundary) {
                std::cerr << "v: " << src(e)
                          << "\t isBoundary: " << src(e).isBoundary()
                          << std::endl;
                std::cerr << "Error: invalid iV " << iV << std::endl;
                exit(1);
            }

            targetLength(iV) =
                geo.edgeLengths[e] *
                exp(0.5 * (uBdy(bIdx[src(e)]) + uBdy(bIdx[dst(e)])));

            T(0, iV) = cos(phi);
            T(1, iV) = sin(phi);

            Ntriplets.emplace_back(iV, iV, geo.edgeLengths[e]);

            phi += kBdy(bIdx[dst(e)]);
        }
    }

    SparseMatrix<double> Ninv(nBoundary, nBoundary);
    Ninv.setFromTriplets(std::begin(Ntriplets), std::end(Ntriplets));

    Vector<double> roundedLength =
        targetLength - Ninv * T.transpose() *
                           (T * Ninv * T.transpose()).inverse() * T *
                           targetLength;

    std::array<Vector<double>, 2> bdyPositions{Vector<double>(nBoundary),
                                               Vector<double>(nBoundary)};
    double x = 0, y = 0;
    for (BoundaryLoop b : mesh.boundaryLoops()) {
        for (Edge e : b.adjacentEdges()) {
            size_t iV = bIdx[src(e)];

            bdyPositions[0](iV) = x;
            bdyPositions[1](iV) = y;

            x += roundedLength(iV) * T(0, iV);
            y += roundedLength(iV) * T(1, iV);
        }
    }

    return bdyPositions;
}
