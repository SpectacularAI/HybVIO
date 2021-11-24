#ifndef ODOMETRY_COMMAND_QUEUE_H_
#define ODOMETRY_COMMAND_QUEUE_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>
#include <queue>
#include <map>

class CommandQueue {
public:
    enum Type {NONE, QUIT, POSE, STEP_MODE, LOCK_BIASES, ROTATE, CONDITION_ON_LAST_POSE, PAUSE_CAMERA, ANY_KEY};

    enum StepMode {CONTINUOUS, ODOMETRY, SLAM};

    struct Command {
        Type type;
        int value;
    };

    CommandQueue();

    virtual void keyboardInput(int key);
    virtual void push(Command c);
    virtual Command dequeue();
    virtual bool empty();

    // Waits for any command and returns it. If it's ANY_KEY, it's removed from queue
    virtual Command waitForAnyKey();

    virtual bool isWaiting();
    virtual StepMode getStepMode();

    virtual std::vector<int> getKeys();

private:
    std::condition_variable condition;
    std::queue<Command> queue;
    std::mutex qMutex;
    std::atomic<bool> waiting;
    std::atomic<StepMode> stepMode;
    std::map<int, std::function<void()>> keymap;
};

#endif // ODOMETRY_COMMAND_QUEUE_H_
