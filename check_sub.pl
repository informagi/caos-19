#!/usr/bin/perl -w

use strict;

# Check a covid-TREC submission for various common errors:
#      * extra fields
#      * multiple run tags
#      * missing or extraneous topics
#      * invalid retrieved documents
#      * duplicate retrieved documents in a single topic
#      * too many documents retrieved for a topic
# Messages regarding submission are printed to an error log

# Results input file is in the form
#     topic_num Q0 docid rank sim tag


# Change these variable values to the full path of the file containing
# the set of valid docids and the directory where the error log should be put
my $docno_loc = "./docids-rnd1.txt";
my $errlog_dir = ".";

# If more than MAX_ERRORS errors, then stop processing; something drastically
# wrong with the file.
my $MAX_ERRORS = 25; 
# May return up to MAX_RET documents per topic
my $MAX_RET = 1000;

my @topics = (1..30);

my %valid_ids;			# set of all valid ids in document set
my %numret;                     # number of docs retrieved per topic
my %retrieved;			# set of retrived docs
my $results_file;               # input file to be checked
my $errlog;                     # file name of error log
my ($q0warn, $num_errors);      # flags for errors detected
my $line;                       # current input line
my ($topic,$q0,$docno,$rank,$sim,$tag);
my $line_num;                   # current input line number
my $run_id;
my ($i,$t,$last_i,$have_docnos);

my $usage = "Usage: $0 resultsfile\n";
$results_file = shift or die $usage;

if ( (! -e $docno_loc) || (! open DOCNOS, "<$docno_loc") ) {
    $have_docnos = 0;
}
else {
    while ($line = <DOCNOS>) {
        chomp $line;
        next if ($line =~ /^\s*$/);
        ($docno) = split " ", $line;
        $valid_ids{$docno} =  1;
    }
    close DOCNOS;
    $have_docnos = 1;
}


open RESULTS, "<$results_file" ||
    die "Unable to open results file $results_file: $!\n";

$last_i = -1;
while ( ($i=index($results_file,"/",$last_i+1)) > -1) {
    $last_i = $i;
}
$errlog = $errlog_dir . "/" . substr($results_file,$last_i+1) . ".errlog";
open ERRLOG, ">$errlog" ||
    die "Cannot open error log for writing\n";

foreach $t (@topics) {
    $numret{$t} = 0;
}
$q0warn = 0;
$num_errors = 0;
$line_num = 0;
$run_id = "";

while ($line = <RESULTS>) {
    chomp $line;
    next if ($line =~ /^\s*$/);

    undef $tag;
    my @fields = split " ", $line;
    $line_num++;
	
    if (scalar(@fields) == 6) {
	($topic,$q0,$docno,$rank,$sim,$tag) = @fields;
    } else {
	&error("Wrong number of fields (expecting 6)");
	exit 255;
    }
	
    # make sure runtag is ok
    if ($run_id eq "") {	# first line --- remember tag 
	$run_id = $tag;
	if ($run_id !~ /^[A-Za-z0-9_.-]{1,24}$/) {
	    &error("Run tag `$run_id' is malformed");
	    next;
	}
    }
    else {		       # otherwise just make sure one tag used
	if ($tag ne $run_id) {
	    &error("Run tag inconsistent (`$tag' and `$run_id')");
	    next;
	}
    }
	
    # get topic number
    if (! exists($numret{$topic})) {
	&error("Unknown topic ($topic)");
	$topic = 0;
	next;
    }
	
	
    # make sure second field is "Q0"
    if ($q0 ne "Q0" && $q0 ne "q0" && ! $q0warn) {
	$q0warn = 1;
	&error("Field 2 is `$q0' not `Q0'");
    }
    
    # remove leading 0's from rank (but keep final 0!)
    $rank =~ s/^0*//;
    if (! $rank) {
	$rank = "0";
    }
	
    # make sure rank is an integer (helps protect against swapping rank and sim)
    if ($rank !~ /^[0-9-]+$/) {
	&error("Column 4 (rank) `$rank' must be an integer");
    }
	
    # make sure DOCNO has right format and is not duplicated
    if (exists $retrieved{$topic}{$docno}) {
	&error("Document `$docno' retrieved more than once for topic $topic");
	next;
    }
    else {
	if (exists $valid_ids{$docno} ||
		(! $have_docnos  && $docno =~ /^[a-z0-9]{8,8}$/) ) {
	    $retrieved{$topic}{$docno} = 1;
	}
        else {
	    &error("Invalid docid `$docno'");
	    next;
	}
    }
    $numret{$topic}++;
}



# Do global checks:
#   error if some topic has no (or too many) documents retrieved for it
#   warn if too few documents retrieved for a topic

foreach $t (@topics) {
    if ($numret{$t} == 0) {
        &error("No documents retrieved for topic $t");
    }
    elsif ($numret{$t} > $MAX_RET) {
        &error("Too many documents ($numret{$t}) retrieved for topic $t");
    }
}


print ERRLOG "Finished processing $results_file\n";
close ERRLOG || die "Close failed for error log $errlog: $!\n";
if ($num_errors) {
    exit 255;
}
exit 0;


# print error message, keeping track of total number of errors
sub error {
    my $msg_string = pop(@_);

    print ERRLOG 
	"$0 of $results_file: Error on line $line_num --- $msg_string\n";

    $num_errors++;
    if ($num_errors > $MAX_ERRORS) {
        print ERRLOG "$0 of $results_file: Quit. Too many errors!\n";
        close ERRLOG ||
	    die "Close failed for error log $errlog: $!\n";
	exit 255;
    }
}

