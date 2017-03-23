#!/usr/bin/perl -w

$/=">c";
open IN,"<$ARGV[0]" or die "$!";
<IN>;
while(<IN>){
	chomp;
	@tmp=split(/\n/,$_);
	$na=$tmp[1];
	@line=split(/\s+/,$tmp[2]);
	$out=$na."\n".$line[0]."\t".$line[3];
	$hash{$out}=1;
	@tmp=();@line=();
	$na="";$out="";
	}
	
foreach $key(keys %hash){
	print "$key\n";
	}
