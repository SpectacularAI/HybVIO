#ifndef ODOMETRY_COMMAND_QUEUE_CPP_
#define ODOMETRY_COMMAND_QUEUE_CPP_

#include "command_queue.hpp"
#include "../util/logging.hpp"
#include <queue>
#include <thread>

CommandQueue::CommandQueue() : queue(), waiting(false), stepMode(StepMode::CONTINUOUS) {
    keymap.emplace('q', [this] { push({ .type = Type::QUIT, .value = 0  }); });
    keymap.emplace(27 /* Escape */, [this] { push({ .type = Type::QUIT, .value = 0  }); });
    keymap.emplace('r', [this] { push({ .type = Type::ROTATE, .value = 0 }); });
    for (int key = 48; key <= 57; key++ /* Horizontal number keys 0-9 */) {
        keymap.emplace(key, [this, key] { push({ .type = Type::POSE, .value = key - 48 }); });
    }
    keymap.emplace('a', [this] {
        if (stepMode.load() != StepMode::ODOMETRY) {
            stepMode.store(StepMode::ODOMETRY);
        } else {
            stepMode.store(StepMode::CONTINUOUS);
        }
        push({ .type = Type::ANY_KEY, .value = 0  }); // To resume playback
    });
    keymap.emplace('s', [this] {
        if (stepMode.load() != StepMode::SLAM) {
            // Print log, as there's no other visual feedback for the activation.
            log_info("Enabled SLAM step mode.");
            stepMode.store(StepMode::SLAM);
        } else {
            log_info("Disabled SLAM step mode.");
            stepMode.store(StepMode::CONTINUOUS);
        }
        push({ .type = Type::ANY_KEY, .value = 0  }); // To resume playback
    });
    keymap.emplace('b', [this] { push({ .type = Type::LOCK_BIASES, .value = 0  }); });
    keymap.emplace('c', [this] { push({ .type = Type::CONDITION_ON_LAST_POSE, .value = 0  }); });
    keymap.emplace('p', [this] { push({ .type = Type::PAUSE_CAMERA, .value = 0  }); });
    keymap.emplace(32 /* Space */, [this] { push({ .type = Type::ANY_KEY, .value = 0  }); });
}

void CommandQueue::keyboardInput(int key) {
    if (keymap.count(key)) {
        keymap[key]();
    } else {
        push({ .type = Type::ANY_KEY, .value = 0  });
    }
}

void CommandQueue::push(Command c) {
    {
        std::lock_guard<std::mutex> lock(qMutex);
        queue.push(c);
    }
    condition.notify_one();
}

CommandQueue::Command CommandQueue::dequeue() {
    std::lock_guard<std::mutex> lock(qMutex);
    if (!queue.empty()) {
        Command c = queue.front();
        queue.pop();
        return c;
    } else {
        return { .type = Type::NONE, .value = 0  };
    }
}

bool CommandQueue::empty() {
    std::lock_guard<std::mutex> lock(qMutex);
    return queue.empty();
}

CommandQueue::Command CommandQueue::waitForAnyKey() {
    std::unique_lock<std::mutex> lock(qMutex);
    waiting.store(true);
    condition.wait(lock, [this] { return !queue.empty(); });
    waiting.store(false);
    Command c = queue.front();
    if (c.type == Type::ANY_KEY) {
        queue.pop();
    }
    return c;
}

bool CommandQueue::isWaiting() {
    return waiting.load();
}

CommandQueue::StepMode CommandQueue::getStepMode() {
    return stepMode.load();
}

std::vector<int> CommandQueue::getKeys() {
    std::vector<int> keys;
    for (auto const &element : keymap) {
        keys.push_back(element.first);
    }
    return keys;
}

#endif // ODOMETRY_COMMAND_QUEUE_CPP_
