echo "Total: $(cat Dimer_status_csvs/status_rank_*.csv | grep -c ",converged") / $(cat Dimer_status_csvs/status_rank_*.csv | wc -l)"
for i in {0..15}; do echo "Rank $i: $(grep -c ",converged" Dimer_status_csvs/status_rank_$i.csv) / $(wc -l < Dimer_status_csvs/status_rank_$i.csv)"; done

