// do visual update
bool odometry.visualUpdateEnabled true
// do visual update for every Nth frame
int odometry.visualUpdateForEveryNFrame 1
// max number of visual update attempts per frame. -1 means no limit
// this is actually the number of attempted triangulations or non-hybrid map
// update, which are the heaviest part
int odometry.maxVisualUpdates 20
// max number of successful visual updates per frame. -1 means no limit
int odometry.maxSuccessfulVisualUpdates 5
// apply visual update in batches
bool odometry.batchVisualUpdate false
// maximum size of the visual batch, relative to the odometry state size
// with values <= 1, the batched update should always be more computationally
// efficient than the non-batched one. Note: a max. size matrix is preallocated
// so do not set this to something like 99999. The minimum safe ~0.286 (2/7)
double odometry.batchVisualUpdateMaxSizeMultiplier 1
// minimum length for tracks used in visual update
int odometry.trackMinFrames 4
// Type 1 threshold for rejecting tracks in visual updates (pixels, -1 to disable check)
double odometry.trackRmseThreshold -1
// Type 2 threshold for rejecting tracks in visual updates (pixels, -1 to disable check)
double odometry.trackChiTestOutlierR 1.5
// On each failed RMSE/Chi2 outlier check, grow both thresholds with this factor.
// Resets on the next frame. Helps to use at least some tracks if all are bad.
// (larger than 1 to enable, 1 to disable)
double odometry.trackOutlierThresholdGrowthFactor 1
bool odometry.scoreVisualUpdateTracks true

// Use a faster linear method for multi-view triangulation
bool odometry.useLinearTriangulation false
// Triangulate stereo image pairs independently and combine as Gaussian
// weighted average in multi-view triangulation. This effectively overrides
// useLinearTriangulation with useStereo. Faster than the iterative PIVO method
bool odometry.useIndependentStereoTriangulation false
// Thresholds for iterative triangulation to consider succeeded and converged.
double odometry.triangulationConvergenceThreshold 1e-2
double odometry.triangulationConvergenceR 11.0
double odometry.triangulationRcondThreshold 1e-8
unsigned odometry.triangulationGaussNewtonIterations 10
// minimum accepted triangulation distance
double odometry.triangulationMinDist 0
// maximum accepted distance: should probably be larger than this default in vehicular cases
double odometry.triangulationMaxDist 1e300
// Track points to use in visual updates:
// GAP: The singe oldest one and all new unused ones.
// ALL: All the points (may be slow and bad from theoretical perspective).
// RANDOM: Use `randomTrackSamplingRatio` of randomly selected points and mark them
//         to be ignored for later samples.
enum TrackSampling odometry.trackSampling GAP GAP ALL RANDOM
// With trackSampling=RANDOM, use this fraction of available points and save rest.
// If set to 1, RANDOM becomes like GAP with the difference of not reusing the oldest point.
double odometry.randomTrackSamplingRatio 0.75
// Triangulate all tracks on each frame to get the full point cloud
// Increases computation load but should not change accuracy
bool odometry.fullPointCloud false

// How many IMU samples the odometry output lags on average.
// Too small value may break sample synchronization.
unsigned odometry.sampleSyncLag 15
// How many frames sample sync buffers before using one. Sensible values: 1 or 2.
unsigned odometry.sampleSyncFrameCount 2
// How many frames can be queued in SampleSync
unsigned odometry.sampleSyncFrameBufferSize 10
// Enable frame rate limiter that will start dropping frames if processing can't keep up before hitting sampleSyncFrameBufferSize
bool odometry.sampleSyncSmartFrameRateLimiter false
// camera trail length used in KF
int odometry.cameraTrailLength 20
// how many elements (at the end) of the camera trail are used as non-FIFO memory slots (Towers-of-Hanoi scheme).
// This can be used to extend the length trail (in time), but with a non uniform stride.
int odometry.cameraTrailHanoiLength 3
// Optional strided part of the pose trail before the "Hanoi" part. In this part of the trail, the elements are also
// kept in a FIFO queue, but new elements are accepted only every N-frames
int odometry.cameraTrailStridedLength 0
int odometry.cameraTrailStridedStride 2
// Keep Hanoi form of the pose trail with increasing deltas, even if features are not tracked from some keyframes anymore.
// This makes the pose trail behave more consistently, which may help with certain visualizations
bool odometry.cameraTrailFixedScheme false

// hybrid EKF-SLAM map point count in the state. Set to 0 to disable the hybrid method
int odometry.hybridMapSize 0
// noise scale used in KF. Tweaking this can increase numerical robustness, or
// it can be used as a tweak factor for making the uncertainty estimates of the
// filter more realistic
double odometry.noiseScale 100
// make continuous zupts in the beginning
bool odometry.useDecayingZeroVelocityUpdate false
// blacklist bad tracks from being used again
bool odometry.blacklistTracks true
// visual_r parameter that (inversely) weights the visual updates
double odometry.visualR 0.05
// Pose trail augmentation update weight.
double odometry.augmentR 1e-9
// Prevent xy velocity from growing beyond `pseudoVelocityLimit` using Kalman filter updates.
bool odometry.usePseudoVelocity false
// pseudo velocity update threshold velocity (m/s)
double odometry.pseudoVelocityLimit 1.4
// which value to update velocity towards ('normally' = pseudo_limit)
double odometry.pseudoVelocityTarget 0
// pseudo velocity update noise parameter R
double odometry.pseudoVelocityR 1e-4
// Uncertainty parameter for zero velocity update aka ZUPT. Sets velocity to zero.
double odometry.zuptR 1e-6
// Uncertainty parameter for zero rotation update aka ZRUPT. Sets gyroscope bias to given sample.
double odometry.rotationZuptR 1e-6
// Uncertainty parameter for ZUPT where the variance is scaled by expired time since start of session.
double odometry.initZuptR 1e-4
// is visual ZUPT enabled in odometry
bool odometry.useVisualStationarity true
// Maximum movement of features on screen to consider the device visually stationary.
double tracker.visualStationarityMovementThreshold 3.0
// Minimum fraction of RANSAC-2 inliers on screen to consider the device visually stationary.
double tracker.visualStationarityScoreThreshold 0.95
// How many consecutive non-keyframes (frames that look stationary to tracker)
// are needed to trigger the visual ZUPT
int odometry.visualStationarityFrameCountThreshold 3
// Uncertainty parameter for ZUPT used on visual stationarity
// the smaller, the stronger the update
double odometry.visualZuptR 1e-7

// gravity vector length
// TODO Is this the best estimate for the local value?
double odometry.gravity 9.819

// odometry EKF noise parameters
// all of these are standard deviations, not variances
// initial position uncertainty (standard deviation, meters)
double odometry.noiseInitialPos 1e-5
// initial orientation uncertainty (standard deviation)
// sqrt(1e-4 * 10) = 0.01 * 3.16227766
double odometry.noiseInitialOri 0.0316227766
// initial velocity uncertainty (standard deviation, m/s)
double odometry.noiseInitialVel 0.1
// initial position trail uncertainty (standard deviation, meters)
double odometry.noiseInitialPosTrail 100
// initial orientation trail uncertainty (standard deviation)
// sqrt(10) = 3.16227766
double odometry.noiseInitialOriTrail 3.16227766
// initial BGA uncertainty (standard deviation, 0 to disable)
double odometry.noiseInitialBGA 1e-3
// initial BAA uncertainty (standard deviation, 0 to disable)
double odometry.noiseInitialBAA 1e-6
// initial BAT uncertainty (standard deviation, 0 to disable)
double odometry.noiseInitialBAT 1e-5
// initial SFT uncertainty (standard deviation)
double odometry.noiseInitialSFT 1e-5
// acc process noise (standard deviation)
double odometry.noiseProcessAcc 0.003
// gyro process noise X (standard deviation)
double odometry.noiseProcessGyro 0.00017

// BAA process noise (standard deviation, 0 to disable)
double odometry.noiseProcessBAA 1e-4
// BGA process noise (standard deviation, 0 to disable)
double odometry.noiseProcessBGA 0

// BAA mean reversion rate. Has effect only if noiseProcessBAA > 0. (0 to disable)
double odometry.noiseProcessBAARev 0.1
// BGA mean reversion rate. Has effect only if noiseProcessBGA > 0. (0 to disable)
double odometry.noiseProcessBGARev 0.1

// Esimate IMU-to-camera time shift (SFT) in the EKF.
bool odometry.estimateImuCameraTimeShift true

// Seed for odometry pseudo random number generators
int odometry.rngSeed 0

// If greater than 0, odometry will run in its own thread with this maximum
// number of unprocessed (gyro/synced) samples in the buffer
unsigned odometry.processingQueueSize 0

// Represents a 4x4 homogeneous matrix in column-major format
// NOTE: If the video data has been rotated at some point (cv::rotate or equivalent)
// this matrix needs to change accordingly, e.g., 0,-1,0,-1,0,0,0,0,-1.
// There also exists a conmmand-line parameter: videoRotation=NONE|CW90|CW180|CW270
// that applies the appropriate modifications (see parameters_base.cpp)
// For backwwards compatiblity, also accepts a 3x3 matrix in column-major format.
std::vector<double> odometry.imuToCameraMatrix 1,0,0,0,-1,0,0,0,-1

// The 4x4 IMU to camera matrix for the second camera. If set to "0", the
// First IMU to camera matrix is used to compute this matrix. If set to a
// 4x4 homogeneous matrix, the "stereoCameraTranslation" is ignored.
// If 3x3 or smaller, then "stereoCameraTranslation" is used for the translation
// part
std::vector<double> odometry.secondImuToCameraMatrix 0
// t - translation between two cameras
std::vector<double> odometry.stereoCameraTranslation 0.0075,0.013,-0.0003
// Add this amount to IMU timestamp to get camera timestamp corresponding to same wall clock.
double odometry.imuToCameraShiftSeconds 0
double odometry.secondImuToCameraShiftSeconds 0

// Reset in the beginning until init succeeds
bool odometry.resetUntilInitSucceeds false
// Reset tracking if it fails
bool odometry.resetOnFailedTracking false
// Reset after tracking fails to initialize after LOST_TRACKING for X seconds
double odometry.resetAfterTrackingFailsToInitialize 3.0
// Freeze output pose when tracking fails
bool odometry.freezeOnFailedTracking false
// Required share of "good frames" where visual tracking succeeds
double odometry.goodFramesToTracking 0.75
double odometry.goodFramesToTrackingFailed 0.05
// Time window for calculating share of "good frames"
double odometry.goodFramesTimeWindowSeconds 2.0

// Use odometry poses and camera parameters to predict and guide optical flow computation.
bool tracker.predictOpticalFlow true
// If true, compute optical flow for right frame from previous right frame.
// If false, compute from left frame.
bool tracker.independentStereoOpticalFlow false

// Cutoff beyond which to trust triangulation results.
double tracker.predictOpticalFlowMinTriangulationDistance 3.0

// Reject stereo LK matches that are further than this from the epipolar curve
// (-1 = disabled. in scale units = min dim / 720)
float tracker.maxStereoEpipolarDistance 10

// fps to target with frame subsampling
double tracker.targetFps 30

// Camera focal length in pixels (-1 = set automatically). If set, applies to both fx and fy
// The alternative is settings focalLengthX and focalLengthY separately.
double tracker.focalLength -1

// Focal length X. Do not set both this and focalLength
double tracker.focalLengthX -1
// Focal length Y. Do not set both this and focalLength
double tracker.focalLengthY -1
// principal point X (-1 = set automatically)
double tracker.principalPointX -1.0
// principal point Y (-1 = set automatically)
double tracker.principalPointY -1.0


// second camera focal length in pixels (-1 = set automatically)
double tracker.secondFocalLength -1
double tracker.secondFocalLengthX -1
double tracker.secondFocalLengthY -1
// second camera principal point X (-1 = set automatically)
double tracker.secondPrincipalPointX -1.0
// second camera principal point Y (-1 = set automatically)
double tracker.secondPrincipalPointY -1.0

// Use an equidistant fisheye instead of the pinhole camera model
bool tracker.fisheyeCamera false

// Valid field-of-view angle for the (fisheye) camera, degerees
// This is required since (LK) feature tracking does not generally work near
// the extremely distorted fisheye image boundaries
float tracker.validCameraFov 140

// Non-linear camera calibration parameters. The interpretation depends on the
// camera model (fishey vs pinhole). Set to 0 (= a vector of length 1) for
// no distortion
std::vector<double> tracker.distortionCoeffs 0

// Distortion coefficients for the second camera in stereo mode
std::vector<double> tracker.secondDistortionCoeffs 0

// max number of feature tracks
int tracker.maxTracks 200
// Use a large enough number when collecting statistics (e.g. 50).
// NOTE: must be at least cameraTrailLength+1
int tracker.maxTrackLength 21

// Tracker RANSAC parameters.
bool tracker.useHybridRansac true
// maximum iterations for custom RANSAC implementation
int tracker.ransacMaxIters 75
// Logic for deciding whether to use RANSAC2 or RANSAC5.
double tracker.ransac2InliersToSkipRansac5 0.9
// Logic for deciding whether to use RANSAC2 or RANSAC5.
double tracker.ransac2InliersOverRansac5Needed 0.9
// threshold parameter for RANSAC2 (in scale units = min dim / 720)
double tracker.ransac2Threshold 4.0
// threshold parameter for RANSAC5
double tracker.ransac5Threshold 2.0
// probability parameter for RANSAC5
double tracker.ransac5Prob 0.999
// minimum inlier fraction threshold for RANSAC2 and RANSAC5
double tracker.ransacMinInlierFraction 0.3
// Theia RANSAC5 parameters
bool tracker.useTheiaRansac5 false
double tracker.theiaRansac5ErrorThresh 5e-5
double tracker.theiaRansac5FailureProbability 1e-4
int tracker.theiaRansac5MaxIterations 500
int tracker.theiaRansac5MinIterations 50
bool tracker.theiaRansac5UseMle true
// Theia RANSAC3 parameters
bool tracker.useRansac3 true
double tracker.ransac3ErrorThresh 1e-4
double tracker.ransac3FailureProbability 1e-4
int tracker.ransac3MaxIterations 500
int tracker.ransac3MinIterations 50
bool tracker.ransac3UseMle true
// "stereo upright 2p" parameters
bool tracker.useStereoUpright2p false
double tracker.ransacStereoUpright2pErrorThresh 1e-4
double tracker.ransacStereoUpright2pFailureProbability 1e-4
int tracker.ransacStereoUpright2pMaxIterations 500
int tracker.ransacStereoUpright2pMinIterations 50
bool tracker.ransacStereoUpright2pUseMle true
// Seed for tracker ransac pipeline pseudo random number generator
int tracker.ransacRngSeed 4649

// feature selection mask radius as a fraction of the smaller image dimension
double tracker.relativeMaskRadius 0.0667 // ~ 1 / 15

// feature detector options: GPU-GFTT, FAST, GFTT
std::string tracker.featureDetector "GPU-GFTT"

// GFTT parameters
// GFFT quality level
double tracker.gfttQualityLevel 0.01
// GFFT min distance (in scale units = smaller dim / 720)
double tracker.gfttMinDistance 50
// GFFT block size
int tracker.gfttBlockSize 3
// GFFT K parameter (only used by Legacy & Harris detectorss?)
double tracker.gfttK 0.04
// Reject features with too low corner response. Unlike gfttQualityLevel, this
// is not relative to any "max response" but in similar units as pyrLKMinEigThreshold
float tracker.gfttMinResponse 1e-3

// Corner sub-pixel refinement parameters
// cornerSubPix window size
int tracker.subPixWindowSize 10
// cornerSubPix max iterations
int tracker.subPixMaxIter 20
// cornerSubPix epsilon (termination criterion)
double tracker.subPixEpsilon 0.03

// Pyramidal LK parameters
// pyramidal LK: max level
int tracker.pyrLKMaxLevel 3
// pyramidal LK: window size
int tracker.pyrLKWindowSize 31
// pyramidal LK: max iterations
int tracker.pyrLKMaxIter 20
// pyramidal LK: epsilon (termination criterion)
double tracker.pyrLKEpsilon 0.03
// pyramidal LK: min eigenvalue threshold
double tracker.pyrLKMinEigThreshold 0.001

int tracker.displayMaxTrackLength 10

// stereo mode parameters
bool tracker.useStereo false
int tracker.leftCameraId 0 // ignored if useStereo = false
int tracker.rightCameraId 1 // ignored if useStereo = false

double tracker.partOfImageToDetectFeatures 1

// enable image rectification
bool tracker.useRectification false
float tracker.rectificationZoom 1.0
bool tracker.computeDenseStereoDepth false
bool tracker.computeStereoPointCloud false

// The stride using which the depth image is traversed/downsampled (applies to both x and y coordinates).
// Usually a good idea to set this to something like 5-20
unsigned tracker.stereoPointCloudStride 5

// enable SLAM (also remember the USE_SLAM CMake flag)
bool slam.useSlam false
bool slam.slamThread true
unsigned slam.maxKeypoints 1000
// minimum number of map point matches for a loop closure
unsigned slam.minLoopClosureFeatureMatches 6
unsigned slam.loopClosureRansacMinInliers 5
unsigned slam.loopClosureRansacIterations 100
// Require tringulation for features before counting them for loop closures
bool slam.requireTringulationForLoopClosures false
// Threshold for matching features based on descriptors. In interval [0, 1], smaller is more strict.
double slam.loopClosureFeatureMatchLoweRatio 0.7
// Maximum accumulated drift in meters per second allowed before loop closure will be rejected
double slam.maximumDriftMetersPerSecond 0.1
// Maximum accumulated drift in radians per second allowed before loop closure will be rejected
double slam.maximumDriftRadiansPerSecond 0.01
// Maximum accumulated drift in meters per meters traveled allowed before loop closure will be rejected
double slam.maximumDriftMetersPerTraveled 0.1
// Maximum accumulated drift in radians per meters traveled allowed before loop closure will be rejected
double slam.maximumDriftRadiansPerTraveled 0.01
bool slam.loopClosureRansacFixScale true
double slam.loopClosureInlierThreshold 0.02 // relative to focal length
// In addition to detecting loop closures, use them to correct the map.
bool slam.applyLoopClosures false
// Do local bundle adjustment on new keyframe.
bool slam.applyLocalBundleAdjustment true
// reprojection error threshold, relative to image size
float slam.relativeReprojectionErrorThreshold 0.02
// was hardcoded as 0.2 in OpenVSLAM. The original Japanese commnt said that
// was equivalent to 2 pixesls on a 90deg FOV camera with resolution 900px.
float slam.epipolarCheckThresholdDegrees 2.0
// minimum number of observation from different keyframes until a map point
// is accepted for (local) Bundle Adjustment
unsigned slam.minObservationsForBA 3
// Don't cull map points younger than this, so they have enough time to gather observations
double slam.minMapPointCullingAge 0.4
// skip BA unless these criteria are met
unsigned slam.minKeyframesInBA 3
unsigned slam.minVisibleMapPointsInCurrentFrameBA 100
unsigned slam.minVisibleMapPointsInNeighborhoodBA 150
unsigned slam.minVisibleMapPointsInForNonKeyframeBA 50
bool slam.nonKeyFramePoseAdjustment true

// Number of keyframes searched over in most SLAM tasks, excluding BA.
int slam.adjacentSpaceSize 20
// Number of keyframes considered in local BA.
int slam.localBAProblemSize 20
// Number of keyframes considered in local BA after a loop closure.
int slam.loopClosureLocalBAProblemSize 40

unsigned slam.globalBAIterations 20
unsigned slam.poseBAIterations 5
// How much to rely on odometry poses in SLAM. Used as a prior for the relative
// pose between consecutive keyframes in bundle adjustment
float slam.odometryPriorStrengthPosition 500
float slam.odometryPriorStrengthRotation 5000
bool slam.odometryPriorSimpleUncertainty false
bool slam.odometryPriorFixed true
// Number of covisible map points needed before keyframes are considered
// neighbours in the SLAM pose graph
unsigned slam.minNeighbourCovisiblitities 10
// add keyframe every N frames
unsigned slam.keyframeCandidateInterval 8
// When enabled, slam is split into light frontend and heavy backend, and slam map is copied from backend to frontend periodically
bool slam.useFrontendSlam false
// Copies SLAM map from backend to frontend every N SLAM frames... which typically
// occur every few frames, depending other settings
unsigned slam.copySlamMapEveryNSlamFrames 2
// Delays backend slam processing by N frames and uses latest odometry pose trail information
unsigned slam.backendProcessDelay 0
// Copy only partial map to frontend
bool slam.copyPartialMapToFrontend true
// When true slam map is copied at fixed frames to ensure deterministic outcome
bool slam.deterministicSlamMapCopy true
// how many times keyframeCandidateInterval we delay the slam calculations to get better pose trail from odometry, keyframeCandidateInterval=4 delayIntervalMultiplier=2 => 8 frames
// Crashes unless cameraTrailLength > keyframeCandidateInterval * (delayIntervalMultiplier + 1)
int slam.delayIntervalMultiplier 1
// Do not let the PIVO -> SLAM transformations tilt the z-axis
bool slam.removeOdometryTransformZAxisTilt true
// When enabled, uses delta between odometry posetrail for odometry priors
bool slam.useOdometryPoseTrailDelta false
// Use variable-length odometry deltas
bool slam.useVariableLengthDeltas false
// skip keyframe decision logic (always true)
bool slam.keyframeDecisionAlways false
// minimum time between two accepted keyframes
double slam.keyframeDecisionMinIntervalSeconds 0.2584
// if less than this many map points in common (ratio), add keyframe
float slam.keyframeDecisionCovisibilityRatio 0.7
// if at moved at least this much since the last keyframe (meters), add keyframe
float slam.keyframeDecisionDistanceThreshold 0.15
float slam.keyframeCullMaxCriticalRatio 0.15
bool slam.keyframeCullEnabled true
bool slam.cullMapPoints true
// path to the ORB vocabulary file that has to be loaded on startup
std::string slam.vocabularyPath "../data/orb_vocab.dbow2"
// Ratio for specifying min number of inliers required in getBowSimilar
// inliers_required = bowMinInCommonRatio * max_inliers
float slam.bowMinInCommonRatio 0.3
// Ratio for specifying min score required in getBowSimilar
// score_required = bowScoreRatio * best_score
float slam.bowScoreRatio 0.5

// Minimum allowed triangulation angles in SLAM for two and many-view cases
float slam.minTriangulationAngleTwoObs 1.0
float slam.minTriangulationAngleMultipleObs 2.0

// Move part of the map rigidly before graph optimization when doing a loop closure.
bool slam.loopClosureRigidTransform false

// Apply global Bundle Adjustment after a loop closure
bool slam.globalBAAfterLoop false

// Orb feature parameters
unsigned slam.orbScaleLevels 8
float slam.orbScaleFactor 1.2
unsigned slam.orbInitialFastThreshold 20
unsigned slam.orbMinFastThreshold 7
unsigned slam.orbLkTrackLevel 2
bool slam.orbExtraKeyPoints true

bool slam.useGpuImagePyramid false
// valid choices for the SLAM ORB feature detector
// * all choices for tracker.featureDetector
// * empty string means "same as tracker.featureDetector"
std::string slam.slamFeatureDetector ""

// TODO The following are essentially command line parameters, but are here for technical reasons.

// * SLAM code doesn't have access to the command line parameters.
// Width of keyframe ASCII visualization in terminal characters.
int slam.kfAsciiWidth 200
// Enable keyframe ASCII visualization for local BA.
bool slam.kfAsciiBA false
// Enable keyframe ASCII visualization for "adjacent" keyframes.
bool slam.kfAsciiAdjacent false

// * Tracker code doesn't have access to the command line parameters.
// Copy feature positions to enable an optical flow visualization. Automatically enabled by
// the command line parameters `-flow` and `-flow2` and meaningless without it.
enum OpticalFlowVisualization tracker.saveOpticalFlow NONE NONE PREDICT COMPARE FAILURES
// Save data for stereo epipolar visualization, set via cmd parameter `displayStereoEpipolarCurves`.
enum StereoEpipolarVisualization tracker.saveStereoEpipolar NONE NONE TRACKED DETECTED FAILED

// Number of clockwise 90 degree rotations to display window. This is also
// set in the stored parameters of benchmark sessions. Therefore not set in
// cmd_parameter_definition.json
int odometry.rot 1
// tolerate missing frames in data JSONL. Command-line-only flag
// but useful to set in parameters.txt and hence defined here
bool odometry.allowSkippedFrames false

// If true, output first camera pose instead of IMU pose.
bool odometry.outputCameraPose false

// Output extra data such as biases in JSONL produced by `-outputPath`.
bool odometry.outputJsonExtras false
bool odometry.outputJsonPoseTrail false

// Process input video using ffmpeg. Set here to allow use in `parameters.txt` and in benchmarks.
bool tracker.ffmpeg false
// Video filters to apply to input video. Requires -ffmpeg.
std::string tracker.vf ""
// use own threads for video reading (faster and utilizes more CPU cores)
bool tracker.videoReaderThreads true
// convert video to grayscale in VideoReader threads (can increase performance)
bool tracker.convertVideoToGray false
// Match grayscale pixel intensities of stereo frames to ease visual tracking between them.
bool tracker.matchStereoIntensities false
double tracker.matchSuccessiveIntensities 0.0

// Output buffer size for compensating for uneven frame processing rate
// not really a part of odometry but rather API
double odometry.targetOutputDelaySeconds 0

// Print statistics about how tracker tracks are processed by the odometry.
bool odometry.printVisualUpdateStats false
// Print statistics about attempted loop closures.
bool slam.printLoopCloserStats false
// Print statistics about bundle adjustments.
bool slam.printBaStats false
