// Deterministic synthetic ride-trip CSV generator used by the benchmarks.
//
//   vecq-gen <rows> [seed] > trips.csv
//
// Columns: trip_id, vendor, pickup_hour, passenger_count (1% NULL),
// distance, fare, tip, payment, zone (260 values).

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: vecq-gen <rows> [seed]\n");
    return 2;
  }
  const long long rows = std::atoll(argv[1]);
  const unsigned seed = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 42u;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::exponential_distribution<double> dist_km(1.0 / 3.2);

  const char* vendors[] = {"acme", "bolt", "cabco", "dash"};
  const char* payments[] = {"card", "cash", "app", "voucher", "dispute"};
  const double pay_w[] = {0.62, 0.20, 0.15, 0.02, 0.01};

  static char buf[1 << 16];
  std::setvbuf(stdout, buf, _IOFBF, sizeof buf);
  std::printf("trip_id,vendor,pickup_hour,passenger_count,distance,fare,tip,payment,zone\n");
  for (long long i = 0; i < rows; ++i) {
    const char* vendor = vendors[rng() % 4];
    // Bimodal-ish hour distribution with rush peaks.
    int hour = static_cast<int>(rng() % 24);
    if (u(rng) < 0.3) hour = (u(rng) < 0.5 ? 8 : 18) + static_cast<int>(rng() % 2);
    double d = std::min(80.0, dist_km(rng) + 0.3);
    double fare = 3.5 + 1.75 * d + 0.4 * static_cast<double>(rng() % 20);
    double p = u(rng), acc = 0;
    int pay = 0;
    for (int k = 0; k < 5; ++k) {
      acc += pay_w[k];
      if (p <= acc) {
        pay = k;
        break;
      }
    }
    double tip = pay == 1 ? 0.0 : fare * (0.05 + 0.2 * u(rng));
    int zone = static_cast<int>(
        std::min<uint64_t>(259, static_cast<uint64_t>(std::fabs(std::normal_distribution<double>(130, 60)(rng)))));
    std::printf("%lld,%s,%d,", i + 1, vendor, hour);
    if (rng() % 100 != 0) std::printf("%d", 1 + static_cast<int>(rng() % 6) / (u(rng) < 0.7 ? 6 : 1));
    std::printf(",%.2f,%.2f,%.2f,%s,Z%03d\n", d, fare, tip, payments[pay], zone);
  }
  return 0;
}
