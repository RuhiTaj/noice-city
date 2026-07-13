#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    uint32_t size = 0;
    if (_NSGetExecutablePath(NULL, &size) != -1 || size == 0) {
        fputs("Noice City could not locate its application bundle.\n", stderr);
        return 70;
    }

    char *executable = calloc((size_t)size + 1, 1);
    if (executable == NULL || _NSGetExecutablePath(executable, &size) != 0) {
        free(executable);
        fputs("Noice City could not resolve its executable path.\n", stderr);
        return 70;
    }

    char *separator = strrchr(executable, '/');
    if (separator == NULL) {
        free(executable);
        return 70;
    }
    *separator = '\0'; /* Contents/MacOS */
    separator = strrchr(executable, '/');
    if (separator == NULL) {
        free(executable);
        return 70;
    }
    *separator = '\0'; /* Contents */

    const char suffix[] = "/Resources/NoiceCity.launcher";
    size_t helper_size = strlen(executable) + sizeof(suffix);
    char *helper = malloc(helper_size);
    if (helper == NULL) {
        free(executable);
        return 70;
    }
    (void)snprintf(helper, helper_size, "%s%s", executable, suffix);

    execl("/bin/bash", "bash", helper, (char *)NULL);
    perror("Noice City could not start its local service");
    free(helper);
    free(executable);
    return 70;
}
