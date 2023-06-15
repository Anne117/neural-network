#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "NetWork.h"
#include "ActivateFunction.h"

TEST_CASE("ReLUActivation::useDer returns correct derivative value")
{
    ReLUActivation activation;

    double negative_value = -0.5;
    double expected_derivative_negative = 0.01;
    double derivative_negative = activation.useDer(negative_value);
    CHECK(derivative_negative == expected_derivative_negative);

    double positive_value = 0.5;
    double expected_derivative_positive = 1.0;
    double derivative_positive = activation.useDer(positive_value);
    CHECK(derivative_positive == expected_derivative_positive);

    double invalid_value = 1.5;
    double expected_derivative_invalid = 0.01;
    double derivative_invalid = activation.useDer(invalid_value);
    CHECK(derivative_invalid == expected_derivative_invalid);
}

TEST_CASE("ReLUActivation::useDer throws exception for NaN value")
{
    ReLUActivation activation;

    double nan_value = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(activation.useDer(nan_value), std::exception);
}

TEST_CASE("ReLUActivation::useDer throws exception for infinity value")
{
    ReLUActivation activation;

    double infinity_value = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(activation.useDer(infinity_value), std::exception);
}

TEST_CASE("SigmoidActivation::useDer returns correct derivative value")
{
    SigmoidActivation activation;

    double positive_value = 0.5;
    double expected_derivative_positive = 0.62245;
    double derivative_positive = activation.useDer(positive_value);
    CHECK(derivative_positive == doctest::Approx(expected_derivative_positive).epsilon(0.0001));

    double negative_value = -0.5;
    double expected_derivative_negative = 0.377541;
    double derivative_negative = activation.useDer(negative_value);
    CHECK(derivative_negative == doctest::Approx(expected_derivative_negative).epsilon(0.0001));

    double zero_value = 0.0;
    double expected_derivative_zero = 0.5;
    double derivative_zero = activation.useDer(zero_value);
    CHECK(derivative_zero == doctest::Approx(expected_derivative_zero).epsilon(0.0001));
}
TEST_CASE("NetWork::SearchMaxIndex returns correct max index")
{
    NetWork network;

    std::vector<double> values1 = {1.5, 2.7, 3.2, 2.1};
    std::vector<double> values2 = {0.5, 0.6, 0.1};
    std::vector<double> values3 = {-1.0, -2.0, -0.5, -0.2, -1.5};
    int max_index1 = network.SearchMaxIndex(values1);
    CHECK(max_index1 == 2);

    int max_index2 = network.SearchMaxIndex(values2);
    CHECK(max_index2 == 1);

    int max_index3 = network.SearchMaxIndex(values3);
    CHECK(max_index3 == 3);
}
