#ifndef ODOMETRY_UTIL_H_
#define ODOMETRY_UTIL_H_

#include <Eigen/Dense>
#include <Eigen/StdVector>

// Fixed-size vectorizable Eigen types need to use special allocators with STL
// containers. See <https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html>.
// Note that this is a very subtle source of bugs, which mostly manifests on
// 32-bit platforms, With normal allocators, you are at the mercy of the
// platform to allocate the start of the array to a 16-byte-aligned address,
// which is only (typically?) guaranteed on 64-bit platforms.
using vecVector2d = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >;
using vecVector3d = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >;
using vecVector3f = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> >;
using vecVector4d = std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> >;
using vecMatrix3d = std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> >;
using vecMatrix4f = std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >;

namespace odometry {

// Chi-square cdf inverse for probability 95% and degrees of freedom by index. Computed in Matlab as `chi2inv(0.95, 0:99)`.
const std::vector<double> chi2inv95 = { 0, 3.841458820694124,5.991464547107981,7.814727903251177,9.487729036781154,11.07049769351635,12.59158724374398,14.06714044934017,15.50731305586545,16.91897760462044,18.30703805327515,19.67513757268249,21.02606981748306,22.36203249482694,23.68479130484057,24.99579013972863,26.29622760486423,27.58711163827532,28.86929943039263,30.14352720564617,31.41043284423091,32.6705733409173,33.92443847144379,35.17246162690805,36.41502850180734,37.65248413348279,38.88513865983008,40.11327206941362,41.33713815142738,42.55696780429268,43.77297182574219,44.98534328036513,46.19425952027845,47.39988391908087,48.60236736729416,49.80184956820188,50.9984601657106,52.19231973010289,53.38354062296929,54.57222775894174,55.75847927888704,56.94238714682412,58.12403768086811,59.30351202689982,60.48088658233645,61.65623337627959,62.8296204114082,64.00111197221806,65.17076890356991,66.33864886296891,67.50480654954127,68.66929391228581,69.8321603398482,70.99345283378231,72.15321616702313,73.31149302908341,74.46832415930936,75.62374846937614,76.77780315606155,77.93052380523042,79.08194448784879,80.23209784876281,81.38101518889917,82.52872654147185,83.67526074272101,84.82064549765674,85.96490744123091,87.1080721953219,88.25016442187419,89.39120787250798,90.53122543488072,91.67023917605469,92.80827038310771,93.94533960119232,95.0814666692433,96.21667075350355,97.35097037903265,98.48438345934034,99.61692732428394,100.74861874635,101.8794739654358,103.0095087122262,104.1387382302737,105.2671772968603,106.3948402427226,107.5217409707194,108.6478929735077,109.7733093502879,110.8980028226843,112.0219857498078,113.1452701425551,114.2678676771939,115.389789708267,116.5110472808736,117.6316511423454,118.7516117533672,119.8709392985671,120.9896436966095,122.1077346098195,123.2252214533618,124.3421134040042,125.4584194084827,126.5741481914943,127.6893082633383,128.8039079272179,129.9179552862288,131.0314582500484,132.1444245413365,133.2568617018678,134.3687770984111,135.4801779283594,136.5910712251353,137.7014638633705,138.8113625638844,139.9207738984558,141.0297042944099,142.1381600390264,143.2461472837745,144.3536720483855,145.4607402247647,146.5673575807672,147.6735297638178,148.7792623044041,149.8845606194415,150.989430015505,152.0938756919582,153.1979027439567,154.3015161653504,155.4047208514822,156.5075216018851,157.60992312289,158.711930030134,159.8135468509983,160.9147780269433,162.0156279157815,163.1161007938603,164.2162008581853,165.3159322284586,166.4152989490638,167.5143049909776,168.61295425362,169.7112505666489,170.8091976916949,171.9067993240401,173.0040590942451,174.1009805697265,175.1975672562818,176.2938225995705,177.389749986549,178.4853527468599,179.5806341541812,180.6755974275342,181.7702457325557,182.8645821827234,183.9586098405564,185.0523317187726,186.145750781418,187.2388699449535,188.3316920793298,189.4242200090044,190.5164565139577,191.6084043306628,192.7000661530269,193.7914446333243,194.8825423830799,195.9733619739492,197.0639059385607,198.1541767713417,199.2441769293169,200.3339088328969,201.4233748666301,202.5125773799402,203.6015186878585,204.6902010717094,205.7786267798057,206.866798028108,207.9547170008703,209.042385851279,210.1298067020601,211.2169816460856,212.3039127469525,213.3906020395573,214.4770515306508,215.5632631993768,216.6492389978069,217.7349808514488,218.8204906597564,219.9057702966149,220.9908216108282,222.0756464265754,223.1602465438775,224.2446237390357,225.3287797650733,226.4127163521544,227.4964352080022,228.5799380183052,229.6632264471097,230.7463021372105,231.8291667105274,232.9118217684764,233.9942688923248 };
namespace util {

Eigen::Matrix3d quat2rmat(const Eigen::Vector4d& q);
Eigen::Matrix3d quat2rmat_d(const Eigen::Vector4d& q, Eigen::Matrix3d(&dR)[4]);
Eigen::Vector4d rmat2quat(const Eigen::Matrix3d& R);
/**
 * Remove any Z-axis tilt from the given rotation matrix, that is,
 * extract the XY-rotation part of the matrix. The input matrix is assumed
 * to be orthogonal (i.e., a rotation matrix)
 */
Eigen::Matrix3d removeRotationMatrixZTilt(const Eigen::Matrix3d &origR);

/**
 * Replace camera orientation in a poseCW (World-to-Camera) matrix but
 * keep orientation.
 */
Eigen::Matrix4d replacePoseOrientationKeepPosition(const Eigen::Matrix4d &origPoseCW, const Eigen::Matrix3d &newOrientationCW);

double sgn(double val);
double stdev(const Eigen::Ref<const Eigen::MatrixXd>& v);
double cond(const Eigen::MatrixXd& A);
double rcond(const Eigen::MatrixXd& A);
double rcond_ldlt(const Eigen::MatrixXd& A);
Eigen::MatrixXd cov2corr(const Eigen::MatrixXd& P);

/** Convert world-to-camera 4d matrix to odometry's (IMU) coordinates. */
void toOdometryPose(
    const Eigen::Matrix4d &poseMatrixWorldToCamera,
    Eigen::Vector3d &position,
    Eigen::Vector4d &orientation,
    const Eigen::Matrix4d &imuToCamera);

/** Convert odometry's (IMU) coordinates to world-to-camera 4d matrix. */
Eigen::Matrix4d toWorldToCamera(
    const Eigen::Vector3d &position,
    const Eigen::Vector4d &orientation,
    const Eigen::Matrix4d &imuToCamera);

/**
 * Convert to camera-to-world matrix. Should be the same as calling
 * .inverse() on the result of toCameraToWorld. However, if the orientaion
 * quaternion is not valid / properly normalized, the results may differ.
 */
Eigen::Matrix4d toCameraToWorld(
    const Eigen::Vector3d &position,
    const Eigen::Vector4d &orientation,
    const Eigen::Matrix4d &imuToCamera);

/**
 * Transform a 3D point [x, y, z] by a homogeneous 4x4 matrix by converting the
 * point to homogeneous coordinates [x, y, z, 1] and back
 */
template<class T>
inline Eigen::Vector<T, 3> transformVec3ByMat4(
    const Eigen::Matrix<T, 4, 4> &mat,
    const Eigen::Vector<T, 3> &vec
) {
    Eigen::Vector<T, 4> vHomog;
    vHomog.template segment<3>(0) = vec;
    vHomog(3) = 1;
    return (mat * vHomog).template segment<3>(0);
}

/**
 * Convert a flat vector into a homogeneous 4x4 matrix. The input format
 * is a flexible for convenience and backwards compatibility. If you want
 * to be most rigorous, use a list of 16 elements in column-major order
 */
inline Eigen::Matrix4d vec2matrix(const std::vector<double> &v) {
    // automatically extract a homogeneous 4x4 matrix
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    switch (v.size()) {
        case 3: // vector of size 3 -> diagonal matrix
            m.diagonal() << v[0], v[1], v[2];
            break;
        case 3*3: // rotation only
            m.topLeftCorner<3,3>() = Eigen::Map<Eigen::Matrix3d>(const_cast<double*>(v.data()));
            break;
        case 4*4: // rotation and translation
            m = Eigen::Map<Eigen::Matrix4d>(const_cast<double*>(v.data()));
            break;
        default:
            assert(false && "invalid list size");
    }
    return m;
}

template<class T>
class CircularBuffer {
public:
	CircularBuffer(size_t size) : mBuffer(size), mSize(size) {};
    CircularBuffer(size_t size, T value) : mBuffer(size), mSize(size) {
        for (size_t i = 0; i < size; i++)
            put(value);
    };
    void put(T value) {
        mBuffer[mHead] = value;
        mHead = (mHead + 1) % mSize;
        if (mEntries != mSize) mEntries++;
    }
    T mean() {
        if (mEntries == 0) return (T)0;
        T total = (T)0;
        for (size_t i = 0; i < mEntries; i++) {
            total += mBuffer[i];
        }
        return total / mEntries;
    }
    T head() {
        return mBuffer[mHead - 1 < 0 ? mSize - 1 : mHead - 1];
    }
    T tail() {
        if (mEntries < mSize) {
            return mBuffer[0];
        } else {
            return mBuffer[mHead];
        }
    }
    size_t maxSize() { return mSize; };
    size_t entries() { return mEntries; };
private:
	std::vector<T> mBuffer;
	size_t mHead = 0;
	size_t mEntries = 0;
	const size_t mSize;
};

constexpr size_t THROUGHPUT_WINDOW_SIZE = 15;
class ThroughputCounter {
public:
    ThroughputCounter() : buffer(THROUGHPUT_WINDOW_SIZE) {};
    void put(double t);
    float throughputPerSecond();
private:
    CircularBuffer<double> buffer;
};

} // namespace util
} // namespace odometry

#endif // ODOMETRY_UTIL_H_
