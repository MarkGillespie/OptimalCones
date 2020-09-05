#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "ConePlacer.h"

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "args/args.hxx"
#include "imgui.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

// == Geometry-central data
std::unique_ptr<ManifoldSurfaceMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;
VertexData<double> niceCones;

// Polyscope visualization handle, to quickly add data to the surface
polyscope::SurfaceMesh* psMesh;

void plotSolution(const VertexData<double>& mu, std::string name,
                  bool estimateLambda = false, bool skipBoundary = false) {

    std::vector<Vector3> cones;
    std::vector<double> angles;
    ConePlacer pl(*mesh, *geometry);
    pl.setVerbose(true);
    VertexData<double> u   = pl.computeU(mu);
    VertexData<double> phi = pl.computePhi(u);

    psMesh->addVertexScalarQuantity(name + " mu", mu);
    psMesh->addVertexScalarQuantity(name + " u", u);
    psMesh->addVertexScalarQuantity(name + " phi", phi);

    cerr << "total " << name << " (interior) cone angle: "
         << pl.getInterior(mu.toVector()).lpNorm<1>()
         << "\t L2 energy: " << pl.L2Energy(u) << endl;
    if (estimateLambda) cerr << "Lambda estimates: " << endl;
    geometry->requireVertexDualAreas();
    for (Vertex v : mesh->vertices()) {
        if (abs(mu[v]) > 1e-8 && (!skipBoundary || !v.isBoundary())) {
            cones.push_back(geometry->inputVertexPositions[v]);
            angles.push_back(mu[v]);

            if (estimateLambda) {
                double lambda = std::copysign(phi[v], mu[v]);
                cerr << lambda
                     << "\t Mλ:" << lambda * geometry->vertexDualAreas[v]
                     << "\t M^-1λ: " << lambda / geometry->vertexDualAreas[v]
                     << endl;
            }
        }
    }

    auto psCloud = polyscope::registerPointCloud(name + " cones", cones);
    auto cloudQ  = psCloud->addScalarQuantity("angle", angles);
    // cloudQ->setEnabled(true);
}

void plotNiceSolution(bool estimateLambda = true) {

    if (niceCones.getMesh() != nullptr) {

        plotSolution(niceCones, "nice", estimateLambda);
    }
}

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {
    static float lambda = 0.5;
    ImGui::SliderFloat("lambda", &lambda, 0.f, 1.f, "lambda=%.3f");

    static int iterations = 12;
    ImGui::SliderInt("iterations", &iterations, 1, 20, "iter=%.3f");

    if (ImGui::Button("Place Cones")) {
        ConePlacer pl(*mesh, *geometry);
        pl.setVerbose(true);

        VertexData<double> u, phi, mu;
        std::tie(u, phi, mu) =
            pl.computeOptimalMeasure(lambda / 100, iterations);


        psMesh->addVertexScalarQuantity("u", u);
        psMesh->addVertexScalarQuantity("phi", phi);
        psMesh->addVertexScalarQuantity("mu", mu);

        VertexData<double> muSparse = pl.contractClusters(mu, true);
        psMesh->addVertexScalarQuantity("muSparse", muSparse);

        VertexData<double> sparseU   = pl.computeU(muSparse);
        VertexData<double> sparsePhi = pl.computePhi(sparseU);

        plotSolution(muSparse, "sparse", false, true);
        plotNiceSolution();
    }

    static int nCones = 4;
    ImGui::SliderInt("Number of greedy cones", &nCones, 1, 100, "iter=%.3f");
    static int gsIterations = 12;
    ImGui::SliderInt("Greedy Iterations", &gsIterations, 1, 20, "iter=%.3f");

    if (ImGui::Button("Place Greedy Cones")) {
        GreedyPlacer gpl(*mesh, *geometry);
        VertexData<double> mu = gpl.niceCones(nCones, gsIterations);

        plotSolution(mu, "greedy", false, true);
    }
}

std::map<size_t, double> readCones(std::string filename) {
    std::ifstream coneFile(filename);
    size_t iV;
    double angle;
    std::map<size_t, double> coneAngles;

    while (coneFile >> iV >> angle) {
        coneAngles[iV] = angle;
    }

    return coneAngles;
}

int main(int argc, char** argv) {

    // Configure the argument parser
    args::ArgumentParser parser("Geometry program");
    args::Positional<std::string> inputFilename(parser, "mesh",
                                                "Mesh to be processed.");
    args::Positional<std::string> coneFilename(
        parser, "cones", "List of cones and cone angles.");

    // Parse args
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    std::string filename = "../../meshes/bunny_small.obj";
    // Make sure a mesh name was given
    if (inputFilename) {
        filename = args::get(inputFilename);
    }

    // Initialize polyscope
    polyscope::init();

    // Set the callback function
    polyscope::state::userCallback = myCallback;

    // Load mesh
    std::tie(mesh, geometry) = loadMesh(filename);
    std::cout << "Genus: " << mesh->genus() << std::endl;

    // Rescale to have unit surface area
    double surfaceArea = 0;
    geometry->requireFaceAreas();
    for (Face f : mesh->faces()) surfaceArea += geometry->faceAreas[f];
    double r = sqrt(surfaceArea);

    for (Vertex v : mesh->vertices()) {
        geometry->inputVertexPositions[v] /= r;
    }

    geometry->refreshQuantities();
    surfaceArea = 0;
    for (Face f : mesh->faces()) surfaceArea += geometry->faceAreas[f];
    std::cout << "Surface area: " << surfaceArea << endl;
    std::cout << "nBoundaryLoops: " << mesh->nBoundaryLoops() << endl;

    // Register the mesh with polyscope
    psMesh = polyscope::registerSurfaceMesh(
        "mesh", geometry->inputVertexPositions, mesh->getFaceVertexList(),
        polyscopePermutations(*mesh));

    if (coneFilename) {
        std::map<size_t, double> cones = readCones(args::get(coneFilename));
        niceCones                      = VertexData<double>(*mesh, 0);
        for (auto const& cone : cones) {
            niceCones[mesh->vertex(cone.first)] = cone.second;
        }
        plotNiceSolution();
    }

    // Give control to the polyscope gui
    polyscope::show();

    return EXIT_SUCCESS;
}
