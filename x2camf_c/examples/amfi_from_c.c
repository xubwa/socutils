/*
 * Minimal example of calling libx2camf_c from plain C: AMFI matrix for an
 * oxygen atom in a small uncontracted even-tempered basis.
 *
 * Build (after building the library):
 *   cc amfi_from_c.c -I../c_api -L../build -lx2camf_c -o amfi_from_c
 *   LD_LIBRARY_PATH=../build ./amfi_from_c
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "x2camf_c.h"

int main(void)
{
    /* 6s 3p uncontracted basis, sorted by angular momentum */
    const int nbas = 9;
    int shell[9] = {0, 0, 0, 0, 0, 0, 1, 1, 1};
    double exp_a[9];
    for (int i = 0; i < 6; i++) exp_a[i] = pow(2.0, 8 - i);
    for (int i = 0; i < 3; i++) exp_a[6 + i] = pow(2.0, 4 - i);

    const int nshell = shell[nbas - 1] + 1;
    const int atom_number = 8; /* oxygen */
    const int flavor = (1 << 0) | (1 << 1); /* gaunt + gauge */

    const int n2c = x2camf_n2c(nbas, shell);
    double* amfi = malloc((size_t)n2c * n2c * sizeof(double));

    int ierr = x2camf_amfi(flavor, atom_number, nshell, nbas, /*print*/ 0,
                           shell, exp_a, X2CAMF_SPEED_OF_LIGHT_DEFAULT, amfi);
    if (ierr != X2CAMF_SUCCESS)
    {
        fprintf(stderr, "x2camf_amfi failed: %s\n",
                x2camf_error_message(ierr));
        free(amfi);
        return 1;
    }

    double norm = 0.0;
    for (int i = 0; i < n2c * n2c; i++) norm += amfi[i] * amfi[i];
    printf("n2c = %d, ||amfi||_F = %.12f\n", n2c, sqrt(norm));

    free(amfi);
    return 0;
}
