# CPI-NXTFusion-

### Table of Contents
* [Requirements](#requirements)
* [Data](#data)
* [Usage](#usage)

## <a name="requirements">Requirements</a>
To run the above script you'll need:
* Python, at least version 3.x
* have installed main/default libraries as: numpy,random, etc...
* [RDkit](https://rdkit.org/)
* [NXFusion](https://bitbucket.org/eddiewrc/nxtfusion/src/master/)
All of the above can be installed via pip or conda.

## <a name="data">Data</a>
The data used in the article are available on:
* Liu: https://github.com/masashitsubaki
* Yunan: https://github.com/luoyunan/DTINet
* Davis/Kiba: https://github.com/hkmztrk/DeepDTA
For what concern protein similarity data and drug similarity data
they can be created with the help of [BLAST+](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download) (with blastp command) and 
with [RDkit](https://rdkit.org/) (directly in python). 
For Pfam data generation you can use [pfamscan](https://anaconda.org/bioconda/pfam_scan) (obtainable via pip).

## <a name="usage">Usage</a>
All can be run directly from bash with: python <script_name>
To run the scripts just make sure to allocate the appropriate data-sources in a 'data'
folder in txt/FASTA format in the same directory(of the script).
