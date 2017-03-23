# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 22:23:46 2015

@author: luo
"""

def get_total_file(fi,fo):
    while True:
         s=fi.readline()
         if not s:
             break
         else:
             fo.write(s)

if __name__=='__main__':

   fi1=open('posi_samples.fasta','r')
   fi2=open('nega_samples.fasta','r')
   fo=open('samples.fasta','w')

   get_total_file(fi1,fo)
   get_total_file(fi2,fo)

   fi1.close()
   fi2.close()
   fo.close()

   fi1=open('posi_seqmap_out','r')
   fi2=open('nega_seqmap_out','r')
   fo=open('seqmap_out','w')

   head_line=fi1.readline()
   fo.write(head_line)
   get_total_file(fi1,fo)

   head_line=fi2.readline()
   get_total_file(fi2,fo)

   fi1.close()
   fi2.close()
   fo.close()
