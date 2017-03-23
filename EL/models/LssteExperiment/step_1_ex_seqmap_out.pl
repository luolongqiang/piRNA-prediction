#!/usr/bin/perl -w

open $fh, "<", "$ARGV[0]" or die "$!";
<$fh>;


while(<$fh>){
    chomp;
    @tmp = split;
    $va = $tmp[4]."\t".$tmp[0]."\t".$tmp[2];
    push @{$hash{$tmp[3]}}, $va;
}

foreach $key (keys %hash){
    print ">$key\n";
    for $item ( @{$hash{$key}} ) {
        print "$item\n";
    }
}
