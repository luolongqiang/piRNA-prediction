#Commands for piRNA prediction by using LSSTE feature
#Input files:posi_samples.fasta, nega_samples.fasta, posi_seqmap_out and nega_seqmap_out
#Output files:lssteResults.csv

mkdir -p temp
#step_0
python step_0_get_total_samples_and_seqmap_out.py
#step_1
perl step_1_ex_seqmap_out.pl seqmap_out >step_1_out
#step_2
perl step_2_change_format_for_RNAplex.pl step_1_out
#step_3
perl step_3_RNAplex.pl temp 
#step_4
perl step_4_change_format_for_Triplets.pl step_3_out >step_4_out
#step_5
perl step_5_ex_Triplets_elements.pl samples.fasta step_4_out >step_5_out
#step_6
python step_6_get_lsste_feature.py
#step_7
python step_7_ex_prediction_for_piRNAs.py
