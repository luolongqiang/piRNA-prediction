#!/usr/bin/perl -w

#########################
open SEQ,"<$ARGV[0]" or die "$!";
while(<SEQ>){
	chomp($na=$_);
	chomp($seq=<SEQ>);
	$se{$na}=$seq;
	
	$na="";$seq="";
	}
close SEQ or die "$!";

########################
$/=">";
open OF,"<$ARGV[1]" or die "$!";
<OF>;
while(<OF>){
	chomp;
	@tmp=split(/\n/,$_);
	if(exists $out{$tmp[0]}){
		$out{$tmp[0]}=$out{$tmp[0]}."\n".$tmp[1];
	}else{
		$out{$tmp[0]}=$tmp[1];
		}
		@tmp=();
	}
close OF or die "$!";

#######################
@tsr=(".",")");
@ch=(A,G,C,T);

for(@tsr){
	$sa1=$_;
	for(@tsr){
		$sa2=$_;
		for(@tsr){
			$sa3=$_;
			for(@ch){
				$ca=$_;
				$tt=$sa1.$sa2.$sa3.$ca;
				push(@CHAR,$tt);
				}
			}
		}
	}
@tsr=();@ch=();
$sa1="";$sa2="";$sa3="";$ca="";$tt="";

########################
$/="\n";

foreach $key(keys %out){
	@tmpp=split(/\n/,$out{$key});
	$sen=">".$key;
	$back=$se{$sen};
	
	for(@tmpp){
		@ls=split(/\s+/,$_);
		@ll=split(/\,/,$ls[1]);
		$start=$ll[0]-1;
		$sub_len=$ll[1]-$ll[0]+1;
		$sub_back=substr($back,$start,$sub_len);
		
		for($i=1;$i<=$sub_len-2;$i++){
			$tm=substr($sub_back,$i,1);
			push(@middle,$tm);
			$tm="";
			}
		if($ls[0]=~/\&(.+)$/){
			$str=$1;
		}else{
			last;
			}
		for($j=0;$j<=$sub_len-3;$j++){
			$tp=substr($str,$j,3);
			$rdy=$tp.$middle[$j];
			push(@tez,$rdy);
			$tp="";$rdy="";
			}
		@ls=();@ll=();@middle=();
		$start="";$sub_len="";$sub_back="";$i="";$j="";
		}
		foreach $key1(@tez){
			$hh{$key1}+=1;
			}
		$jun=$#tez+1;
		$sum_ma=$#tmpp+1;
		print "$sen\t";
		foreach $key2(@CHAR){
			if(exists $hh{$key2}){
			$jun_hou=$hh{$key2}/$jun;
			print "$jun_hou\t";
		}else{
			print "0\t";
			}
		}
		print "\n";		
		@tmpp=();@tez=();%hh=();
		$sen="";$back="";$jun="";$sum_ma="";$jun_hou="";
	}
