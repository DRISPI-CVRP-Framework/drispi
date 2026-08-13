# pdflatex + bibtex + glossaries (acronyms + symbols)
$pdf_mode = 1;
$bibtex_use = 2;

add_cus_dep('acn', 'acr', 0, 'makeglossaries');
add_cus_dep('syg', 'syi', 0, 'makeglossaries');
add_cus_dep('glo', 'gls', 0, 'makeglossaries');

sub makeglossaries {
    my ($base_name, $path) = fileparse($_[0]);
    pushd $path;
    my $return = system 'makeglossaries', $base_name;
    popd;
    return $return;
}

push @generated_exts, 'glo', 'gls', 'glg';
push @generated_exts, 'acn', 'acr', 'alg';
push @generated_exts, 'syg', 'syi', 'slg';
push @generated_exts, 'ist', 'glsdefs';
