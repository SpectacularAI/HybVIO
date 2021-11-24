#include "catch2/catch.hpp"
#include "../src/util/allocator.hpp"

#include <iostream>

TEST_CASE( "image_allocator", "[util]" ) {

    struct Thing {
        int &counter;
        Thing(int &counter) : counter(counter) {
            counter++;
        }
        ~Thing() {
            counter--;
        }
    };

    int thingCounter = 0;
    const int INITIAL_CAPACITY = 3;

    auto thingAllocator = [&thingCounter]() {
        return std::make_unique<Thing>(thingCounter);
    };

    util::Allocator<Thing> allocator(thingAllocator, INITIAL_CAPACITY);

    REQUIRE(thingCounter == INITIAL_CAPACITY);
    auto thing1 = allocator.next();
    auto thing2 = allocator.next();
    REQUIRE(thingCounter == INITIAL_CAPACITY);
    {
        auto thing3 = allocator.next();
        REQUIRE(thingCounter == INITIAL_CAPACITY);
    }
    auto thing3 = allocator.next();
    REQUIRE(thingCounter == INITIAL_CAPACITY);

    auto thing4 = allocator.next();
    REQUIRE(thingCounter > INITIAL_CAPACITY);
}
