#include "catch2/catch.hpp"
#include "../src/util/parameter_parser.hpp"

TEST_CASE( "parameter parser", "[parameter_parser]" ) {
    ParameterParser parser;
    parser.add("foo", "123");
    parser.add("bar", "baz");
    parser.add("thing", "1.5");
    parser.add("truething", "true");
    parser.add("falsething", "false");

    REQUIRE(parser.hasKey("foo"));
    REQUIRE(!parser.hasKey("baz"));

    REQUIRE(parser.get<int>("foo") == 123 );
    REQUIRE(parser.get<double>("thing") == 1.5 );
    REQUIRE(parser.get<float>("thing") == 1.5 );
    REQUIRE(parser.get<bool>("truething"));
    REQUIRE(!parser.get<bool>("falsething"));
    REQUIRE(parser.getUnusedKeys().size() == 1);
    REQUIRE(parser.getUnusedKeys().count("bar") == 1);
    REQUIRE_THROWS(parser.throwOnErrors());
    REQUIRE(parser.hasKey("bar"));
    REQUIRE(parser.getUnusedKeys().count("bar") == 1);

    REQUIRE(parser.get<std::string>("bar") == "baz" );
    REQUIRE(parser.getUnusedKeys().size() == 0);
    REQUIRE_NOTHROW(parser.throwOnErrors());

    REQUIRE_THROWS(parser.get<int>("bar"));
    REQUIRE_THROWS(parser.get<bool>("bar"));
    REQUIRE_THROWS(parser.get<std::string>("zzz"));
}

TEST_CASE( "parse delimited parser", "[parameter_parser]" ) {
    ParameterParser parser;
    parser.parseDelimited("foo bar; baz 1.34; qux true");

    REQUIRE(parser.get<std::string>("foo") == "bar" );
    REQUIRE(parser.get<double>("baz") == 1.34 );

    REQUIRE(parser.getUnusedKeys().size() == 1);
    REQUIRE(parser.getUnusedKeys().count("qux") == 1);

}

TEST_CASE( "parse command line valid", "[parameter_parser]" ) {
    const char *argv[] = {
            "name_of_the_program",
            "-name=value1",
            "--foo=123",
            "-empty=",
            "--baz",
            "-bar=45",
    };
    const int argc = 6;

    ParameterParser parser;
    parser.parseCommandLine(argc, const_cast<char**>(argv));
    REQUIRE(parser.get<std::string>("empty") == "" );
    REQUIRE(parser.get<bool>("baz"));
    REQUIRE(parser.get<int>("bar") == 45);

    REQUIRE(parser.getUnusedKeys().size() == 2);
}

TEST_CASE( "parse command line invalid", "[parameter_parser]" ) {
    const char *argv[] = {
            "name_of_the_program",
            "ddfg",
    };
    const int argc = 2;

    ParameterParser parser;
    REQUIRE_THROWS(parser.parseCommandLine(argc, const_cast<char**>(argv)));
}
