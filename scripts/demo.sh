#!/bin/sh
# Runs a few representative queries against a generated CSV.
set -e
VECQ=${1:-build/vecq}
DATA=${2:-bench/data/demo.csv}
run() {
  echo "\$ vecq $DATA \"$1\""
  "$VECQ" --timing "$DATA" "$1"
  echo
}
"$VECQ" --schema "$DATA"
echo
run "SELECT payment, COUNT(*) AS trips, AVG(fare) AS avg_fare, AVG(tip) FROM t GROUP BY payment ORDER BY trips DESC"
run "SELECT zone, SUM(tip) AS tips FROM t WHERE pickup_hour BETWEEN 17 AND 19 AND distance > 2.5 GROUP BY zone ORDER BY tips DESC LIMIT 5"
run "SELECT trip_id, vendor, fare FROM t WHERE fare > 150 AND payment = 'cash' ORDER BY fare DESC LIMIT 3"
