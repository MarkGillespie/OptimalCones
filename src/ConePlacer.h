#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/numerical/linear_solvers.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/utilities.h" // RandomReal

#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include <deque>

using namespace geometrycentral;
using namespace geometrycentral::surface;

using std::cerr;
using std::cout;
using std::endl;

using Eigen::Dynamic;

class ConePlacer {
  public:
    ConePlacer(ManifoldSurfaceMesh& mesh_, VertexPositionGeometry& geo_);

    std::array<VertexData<double>, 3>
    computeOptimalMeasure(double lambda = 1, size_t regularizedSteps = 4);

    VertexData<double> contractClusters(const VertexData<double>& x);

    VertexData<double> computeU(const VertexData<double>& mu);
    VertexData<double> computePhi(const VertexData<double>& u);
    double projErr(const VertexData<double>& mu, const VertexData<double>& phi);
    double L2Energy(const VertexData<double>& u);

    void setVerbose(bool verb);

  private:
    std::array<Vector<double>, 2> computeRegularizedMeasure(Vector<double> u,
                                                            Vector<double> phi,
                                                            double lambda,
                                                            double gamma);

    std::array<VertexData<double>, 3>
    computeOptimalMeasure(Vector<double> u, Vector<double> phi, double lambda);

    Vector<double> residual(const Vector<double>& u, const Vector<double>& phi,
                            const Vector<double>& mu, double lambda);

    Vector<double> regularizedResidual(const Vector<double>& u,
                                       const Vector<double>& phi, double lambda,
                                       double gamma);

    SparseMatrix<double> computeDF(const Vector<double>& u,
                                   const Vector<double>& phi,
                                   const Vector<double>& mu, double lambda);

    SparseMatrix<double> computeRegularizedDF(const Vector<double>& u,
                                              const Vector<double>& phi,
                                              double lambda, double gamma);

    SparseMatrix<double> D(const Vector<double>& x, double lambda);
    std::vector<double> Dvec(const Vector<double>& x, double lambda);

    double muSum(const Vector<double>& mu);
    Vector<double> normalizeMuSum(const Vector<double>& mu);

    Vector<double> P(Vector<double> x, double lambda);

    bool checkSubdifferential(const Vector<double>& mu,
                              const Vector<double>& phi, double lambda);

    double checkPhiIsEnergyGradient(const Vector<double>& mu,
                                    const Vector<double>& phi, double lambda,
                                    double epsilon = 1e-6);

    Vector<double> computePhi(const Vector<double>& u);
    Vector<double> computeU(const Vector<double>& mu);
    Vector<double> projectOutConstant(const Vector<double>& vec);
    double computeDistortionEnergy(const Vector<double>& mu, double lambda);
    double Lagrangian(const Vector<double>& mu, const Vector<double>& u,
                      const Vector<double>& phi);

    std::array<Vector<double>, 2>
    splitInteriorBoundary(const Vector<double>& vec);
    Vector<double> combineInteriorBoundary(const Vector<double>& interior,
                                           const Vector<double>& boundary);
    Vector<double> extendBoundaryByZero(const Vector<double>& boundary);
    Vector<double> extendInteriorByZero(const Vector<double>& interior);
    Vector<double> extendInteriorByZero(const std::vector<double>& interior);
    Vector<double> getInterior(const Vector<double>& vec);
    Vector<double> getBoundary(const Vector<double>& vec);

    ManifoldSurfaceMesh& mesh;
    VertexPositionGeometry& geo;
    VertexData<size_t> vIdx;
    SparseMatrix<double> Lii, Lib, Mii;
    std::unique_ptr<PositiveDefiniteSolver<double>> Liisolver;
    Vector<double> Omegaii;

    Vector<bool> isInterior;
    size_t nInterior, nBoundary, nVertices;

    bool verbose = false;
};

Vector<double> stackVectors(const std::vector<Vector<double>>& vs);

std::vector<Vector<double>> unstackVectors(const Vector<double>& bigV,
                                           const std::vector<size_t>& sizes);
