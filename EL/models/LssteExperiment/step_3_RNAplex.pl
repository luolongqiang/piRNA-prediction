#!/usr/bin/perl -w

opendir DIR,"$ARGV[0]" or die "$!";

@f_n=readdir(DIR);

closedir DIR or die "$!";

for(@f_n){
    if(/^\w+/){
        if(/^target.+/){
            push(@T,$_);
        }else{
            push(@O,$_);
        }
    }else{
        next;
    }
}

for(@O){
    $in=$_;
    for(@T){
        $ta=$_;
        if($ta=~/$in$/){
            $hash{$in}=$ta;
        }
    }
}

foreach $key(keys %hash){
   system("./RNAplex -t ./temp/$hash{$key} -q ./temp/$key >>step_3_out");
}

