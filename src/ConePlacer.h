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

    VertexData<double> contractClusters(VertexData<double>& mu,
                                        bool fancy = false);

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

    Vector<double> P(Vector<double> x, double lambda);

    Vector<double> computePhi(const Vector<double>& u);
    Vector<double> computeU(const Vector<double>& mu);
    double computeDistortionEnergy(const Vector<double>& mu, double lambda);

    std::pair<std::vector<Vertex>, std::vector<double>>
    approximateCluster(const VertexData<double>& mu,
                       const std::vector<Vertex>& cluster);

    ManifoldSurfaceMesh& mesh;
    VertexPositionGeometry& geo;
    VertexData<size_t> vIdx;
    SparseMatrix<double> L, M;
    std::unique_ptr<PositiveDefiniteSolver<double>> Lsolver;
    Vector<double> Omega;

    size_t nVertices;

    bool verbose = false;
};

Vector<double> stackVectors(const std::vector<Vector<double>>& vs);

std::vector<Vector<double>> unstackVectors(const Vector<double>& bigV,
                                           const std::vector<size_t>& sizes);

SparseMatrix<double> speye(size_t n);
// C is a bracket-addressable container containing elements of type T
// F is a function taking in type T and returning type S
// S must be comparable with >
template <typename C, typename F,
          typename T = typename std::remove_reference<
              decltype((*(C*)nullptr)[(size_t)0])>::type,
          typename S = typename std::result_of<F(T)>::type>
inline size_t argmax(const C& container, const F& fn) {
    size_t maxIdx  = 0;
    size_t currIdx = 0;
    S maxVal       = fn(container[0]);
    for (T elem : container) {
        S currVal = fn(elem);
        if (currVal > maxVal) {
            maxIdx = currIdx;
            maxVal = currVal;
        }
        currIdx++;
    }
    return maxIdx;
}

template <typename T>
inline size_t argmax(const VertexData<T>& data) {
    // TODO: directly iterate over data
    Eigen::Matrix<T, Eigen::Dynamic, 1> dataVec = data.toVector();
    size_t currMaxIdx                           = 0;
    T currMaxVal                                = dataVec[0];
    for (size_t iV = 1; iV < (size_t)dataVec.size(); iV++) {
        if (dataVec[iV] > currMaxVal) {
            currMaxIdx = iV;
            currMaxVal = dataVec[iV];
        }
    }
    return currMaxIdx;
}

template <typename T, typename F,
          typename S = typename std::result_of<F(T)>::type>
inline size_t argmax(const VertexData<T>& data, const F& fn) {
    // TODO: directly iterate over data
    Eigen::Matrix<T, Eigen::Dynamic, 1> dataVec = data.toVector();
    size_t currMaxIdx                           = 0;
    S currMaxVal                                = fn(dataVec[0]);
    for (size_t iV = 1; iV < (size_t)dataVec.size(); iV++) {
        if (fn(dataVec[iV]) > currMaxVal) {
            currMaxIdx = iV;
            currMaxVal = fn(dataVec[iV]);
        }
    }
    return currMaxIdx;
}
