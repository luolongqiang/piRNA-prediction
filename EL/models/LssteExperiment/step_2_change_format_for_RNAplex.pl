#!/usr/bin/perl -w

$/=">";
open IN, "<$ARGV[0]" or die "$!";
<IN>;

while(<IN>){
    chomp;
    @tmp=split(/\n/,$_);
    $seq_name=shift(@tmp);
    
    for(@tmp){
        ($seq,$tar_name,$tar_seq)=(split /\s+/,$_);
        $hash{$tar_name}=$tar_seq;
    }
    open F1,">./temp/$seq_name" or die "$!";
    open F2,">./temp/target_$seq_name" or die "$!";
    print F1 ">$seq_name\n$seq\n";
    
    foreach $key(keys %hash){
        print F2 ">$key\n$hash{$key}\n";
    }
    
    @tmp=();$seq_name="";
    $seq="";$tar_name="";
    $tar_seq="";
    %hash=();

}
