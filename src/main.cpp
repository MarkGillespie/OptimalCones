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

void plotNiceSolution() {

    if (niceCones.getMesh() != nullptr) {

        ConePlacer pl(*mesh, *geometry);
        pl.setVerbose(true);
        VertexData<double> niceU   = pl.computeU(niceCones);
        VertexData<double> nicePhi = pl.computePhi(niceU);

        psMesh->addVertexScalarQuantity("Nice mu", niceCones);
        psMesh->addVertexScalarQuantity("Nice u", niceU);
        psMesh->addVertexScalarQuantity("Nice phi", nicePhi);

        cerr << "total nice (interior) cone angle: "
             << niceCones.toVector().lpNorm<1>()
             << "\t L2 energy: " << pl.L2Energy(niceU) << endl;
        std::vector<double> lambdaEstimates;
        for (Vertex v : mesh->vertices()) {
            if (abs(niceCones[v]) > 1e-8) {
                lambdaEstimates.push_back(
                    std::copysign(nicePhi[v], niceCones[v]));
            }
        }

        cerr << "Lambda estimates: " << endl;
        for (double lambda : lambdaEstimates) {
            cerr << lambda << endl;
        }
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

        VertexData<double> muSparse = pl.contractClusters(mu);

        psMesh->addVertexScalarQuantity("u", u);
        psMesh->addVertexScalarQuantity("phi", phi);
        psMesh->addVertexScalarQuantity("mu", mu);
        psMesh->addVertexScalarQuantity("muSparse", muSparse);
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
