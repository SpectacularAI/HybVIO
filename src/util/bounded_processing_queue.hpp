#ifndef UTIL_BOUNDED_PROCESSING_QUEUE_HPP
#define UTIL_BOUNDED_PROCESSING_QUEUE_HPP

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>
#include <accelerated-arrays/future.hpp>
#include "allocator.hpp"

namespace util {
class BoundedProcessingQueue {
public:
    // If zero, all items will be processed instantly
    // NOTE: it's not safe to destroy this object if some enqueue calls are
    // still blocking. Otherwise thread-safe
    BoundedProcessingQueue(size_t maxSize) :
        processor(maxSize > 0 ? accelerated::Processor::createThreadPool(1) : nullptr)
    {
        ringBuffer.resize(maxSize);
    }

    size_t maxSize() const { return ringBuffer.size(); }

    void enqueue(const std::function<void()> &job) {
        enqueuePrivate(job, false);
    }

    void enqueueAndWait(const std::function<void()> &job) {
        enqueuePrivate(job, true);
    }

private:
    void enqueuePrivate(const std::function<void()> &job, bool wait) {
        if (!processor) {
            job();
            return;
        }

        std::unique_lock<std::mutex> lock(m);
        auto &fut = ringBuffer.at(ringBufferIndex);
        if (!fut) {
            fut.reset(new accelerated::Future(processor->enqueue(job)));
        } else {
            fut->wait();
            *fut = processor->enqueue(job);
        }
        size_t curIndex = ringBufferIndex;
        ringBufferIndex = (ringBufferIndex + 1) % ringBuffer.size();
        if (wait) {
            auto fut = ringBuffer.at(curIndex);
            lock.unlock();
            fut->wait();
        }
    }

    std::mutex m;
    std::vector<std::shared_ptr<accelerated::Future>> ringBuffer;
    size_t ringBufferIndex = 0;
    std::unique_ptr<accelerated::Processor> processor;
};

// Not thread-safe: Only access object from one thread
template <class Item> class BoundedInputQueue {
public:
    using Producer = std::function<bool(Item&)>;
    BoundedInputQueue(
        size_t maxSize,
        const Producer &producer,
        const std::function<std::unique_ptr<Item>()> builder = [](){ return std::make_unique<Item>(); })
    :
        queue(maxSize),
        producer(producer),
        allocator(builder, std::max(maxSize, size_t(1)))
    {}

    std::shared_ptr<Item> get() {
        if (ringBuffer.empty()) {
            for (size_t i = 0; i < std::max(queue.maxSize(), size_t(1)); ++i) {
                ringBuffer.push_back(allocator.next());
                startJob(ringBuffer.at(i));
            }
        }

        auto frame = ringBuffer.at(ringBufferIndex);
        ringBuffer.at(ringBufferIndex) = allocator.next();
        startJob(ringBuffer.at(ringBufferIndex));
        ringBufferIndex = (ringBufferIndex + 1) % ringBuffer.size();
        return frame;
    }

private:
    void startJob(std::shared_ptr<Item> item) {
        queue.enqueue([this, item]() {
            // safe to access without mutex since only used from the queue thread
            if (finished) return;
            if (!producer(*item)) {
                finished = true;
            }
        });
    }

    BoundedProcessingQueue queue;
    Producer producer;
    Allocator<Item> allocator;

    std::vector<std::shared_ptr<Item>> ringBuffer;
    int ringBufferIndex = 0;
    bool finished = false;
};
}

#endif
