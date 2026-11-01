# Terminal recording of the vecq demo. Regenerate with: vhs demo.tape
Output docs/demo.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 760
Set TypingSpeed 25ms
Set Theme "Builtin Dark"

Hide
Type "export PS1='$ ' && export PATH=$PWD/build:$PATH && clear" Enter
Show

Type "ls -lh bench/data/trips_10000000.csv | awk '{print $5, $9}'" Enter
Sleep 1s
Type `vecq --timing --threads 9 bench/data/trips_10000000.csv "SELECT payment, COUNT(*) AS trips, AVG(fare), AVG(tip) FROM t GROUP BY payment ORDER BY trips DESC"` Enter
Sleep 3s
Type `vecq --timing --threads 9 bench/data/trips_10000000.csv "SELECT zone, SUM(tip) AS tips FROM t WHERE pickup_hour BETWEEN 17 AND 19 AND distance > 2.5 GROUP BY zone ORDER BY tips DESC LIMIT 5"` Enter
Sleep 4s
