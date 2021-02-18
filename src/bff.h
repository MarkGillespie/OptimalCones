#pragma once

#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/numerical/linear_solvers.h"
#include "geometrycentral/surface/intrinsic_geometry_interface.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

using std::cerr;
using std::cout;
using std::endl;

using Eigen::Dynamic;

class BFF {
  public:
    BFF(ManifoldSurfaceMesh& mesh_, IntrinsicGeometryInterface& geo_);

    VertexData<Vector2> flattenMAD();
    VertexData<Vector2> flattenFromU(const VertexData<double>& uBdy);

    Vector<double> dirichletToNeumann(const Vector<double> uBdy);

    std::array<Vector<double>, 2>
    computeRoundedBoundary(const Vector<double>& uBdy,
                           const Vector<double>& kBdy);

    VertexData<double> u;

  protected:
    ManifoldSurfaceMesh& mesh;
    IntrinsicGeometryInterface& geo;
    VertexData<size_t> vIdx;
    VertexData<int> iIdx, bIdx;

    SparseMatrix<double> L, Lii, Lib, Lbb;
    std::unique_ptr<PositiveDefiniteSolver<double>> Liisolver;
    Vector<double> Omegai, Omegab;

    Vector<bool> isInterior;
    size_t nInterior, nBoundary, nVertices;

    BlockDecompositionResult<double> Ldecomp;
};
