set title 'Visualisation NMI'
set xlabel 'nombre_execution'
set ylabel 'NMI'

# labels
set label "boiling point" at 10, 212

plot 'nmi_data.dat' using ($0+1):1 with linespoints 
set term png
set output "nmi.png"
replot
set term x11