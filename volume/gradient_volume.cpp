#include "gradient_volume.h"
#include <algorithm>
#include <exception>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>
#include <gsl/span>

namespace volume {

// Compute the maximum magnitude from all gradient voxels
static float computeMaxMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::max_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute the minimum magnitude from all gradient voxels
static float computeMinMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::min_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute a gradient volume from a volume
static std::vector<GradientVoxel> computeGradientVolume(const Volume& volume)
{
    const auto dim = volume.dims();

    std::vector<GradientVoxel> out(static_cast<size_t>(dim.x * dim.y * dim.z));
    for (int z = 1; z < dim.z - 1; z++) {
        for (int y = 1; y < dim.y - 1; y++) {
            for (int x = 1; x < dim.x - 1; x++) {
                const float gx = (volume.getVoxel(x + 1, y, z) - volume.getVoxel(x - 1, y, z)) / 2.0f;
                const float gy = (volume.getVoxel(x, y + 1, z) - volume.getVoxel(x, y - 1, z)) / 2.0f;
                const float gz = (volume.getVoxel(x, y, z + 1) - volume.getVoxel(x, y, z - 1)) / 2.0f;

                const glm::vec3 v { gx, gy, gz };
                const size_t index = static_cast<size_t>(x + dim.x * (y + dim.y * z));
                out[index] = GradientVoxel { v, glm::length(v) };
            }
        }
    }
    return out;
}

GradientVolume::GradientVolume(const Volume& volume)
    : m_dim(volume.dims())
    , m_data(computeGradientVolume(volume))
    , m_minMagnitude(computeMinMagnitude(m_data))
    , m_maxMagnitude(computeMaxMagnitude(m_data))
{
}

float GradientVolume::maxMagnitude() const
{
    return m_maxMagnitude;
}

float GradientVolume::minMagnitude() const
{
    return m_minMagnitude;
}

glm::ivec3 GradientVolume::dims() const
{
    return m_dim;
}

// This function returns a gradientVoxel at coord based on the current interpolation mode.
GradientVoxel GradientVolume::getGradientVoxel(const glm::vec3& coord) const
{
    switch (interpolationMode) {
    case InterpolationMode::NearestNeighbour: {
        return getGradientVoxelNN(coord);
    }
    case InterpolationMode::Linear: {
        return getGradientVoxelLinearInterpolate(coord);
    }
    case InterpolationMode::Cubic: {
        // No cubic in this case, linear is good enough for the gradient.
        return getGradientVoxelLinearInterpolate(coord);
    }
    default: {
        throw std::exception();
    }
    };
}

// This function returns the nearest neighbour given a position in the volume given by coord.
// Notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions
GradientVoxel GradientVolume::getGradientVoxelNN(const glm::vec3& coord) const
{
    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord, glm::vec3(m_dim))))
        return { glm::vec3(0.0f), 0.0f };

    auto roundToPositiveInt = [](float f) {
        return static_cast<int>(f + 0.5f);
    };

    return getGradientVoxel(roundToPositiveInt(coord.x), roundToPositiveInt(coord.y), roundToPositiveInt(coord.z));
}

// ======= TODO : IMPLEMENT ========
// Returns the trilinearly interpolated gradinet at the given coordinate.
// Use the linearInterpolate function that you implemented below.
GradientVoxel GradientVolume::getGradientVoxelLinearInterpolate(const glm::vec3& coord) const
{
    // Find the lower corner of the cube which the coordinate fits in
    int x = floor(coord.x);
    int y = floor(coord.y);
    int z = floor(coord.z);

    // Determine the factors for all axes
    float fac_x = coord.x - x;
    float fac_y = coord.y - y;
    float fac_z = coord.z - z;

    // Interpolate for all x-axes given the different sample points
    GradientVoxel r_1 = linearInterpolate(getGradientVoxel(x, y, z), getGradientVoxel(x + 1, y, z), fac_x);
    GradientVoxel r_2 = linearInterpolate(getGradientVoxel(x, y + 1, z), getGradientVoxel(x + 1, y + 1, z), fac_x);
    GradientVoxel r_3 = linearInterpolate(getGradientVoxel(x, y, z + 1), getGradientVoxel(x + 1, y, z + 1), fac_x);
    GradientVoxel r_4 = linearInterpolate(getGradientVoxel(x, y + 1, z + 1), getGradientVoxel(x + 1, y + 1, z + 1), fac_x);

    // Interpolate resulting gradients for the other axes
    GradientVoxel r_5 = linearInterpolate(r_1, r_2, fac_y);
    GradientVoxel r_6 = linearInterpolate(r_3, r_4, fac_y);
    return linearInterpolate(r_5, r_6, fac_z);
}

// ======= TODO : IMPLEMENT ========
// This function should linearly interpolates the value from g0 to g1 given the factor (t).
// At t=0, linearInterpolate should return g0 and at t=1 it returns g1.
GradientVoxel GradientVolume::linearInterpolate(const GradientVoxel& g0, const GradientVoxel& g1, float factor)
{
    glm::vec3 result = (1 - factor) * g0.dir + factor * g1.dir;
    float mag = sqrt(result.x * result.x + result.y * result.y + result.z * result.z);
    return GradientVoxel { result, mag};
}

// This function returns a gradientVoxel without using interpolation
GradientVoxel GradientVolume::getGradientVoxel(int x, int y, int z) const
{
    const size_t i = static_cast<size_t>(x + m_dim.x * (y + m_dim.y * z));
    return m_data[i];
}
}