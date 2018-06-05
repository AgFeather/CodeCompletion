var path = require("path");
var _ = require("lodash");

var pkg = require("./package.json");

module.exports = function (grunt) {
    // Path to the client src
    var srcPath = path.resolve(__dirname, "src");

    // Load grunt modules
    grunt.loadNpmTasks('grunt-hr-builder');
    grunt.loadNpmTasks('grunt-exec');
    grunt.loadNpmTasks('grunt-contrib-clean');
    grunt.loadNpmTasks('grunt-contrib-copy');
    grunt.loadNpmTasks('grunt-node-webkit-builder');
    grunt.loadNpmTasks('grunt-github-releaser');
    grunt.loadNpmTasks("grunt-bower-install-simple");

    var NW_VERSION = "0.10.3";

    // Init GRUNT configuraton
    grunt.initConfig({
        pkg: pkg,
        "bower-install-simple": {
            options: {
                color:       true,
                production:  false,
                directory:   "src/vendors"
            }
        },
        'hr': {
            app: {
                "source": path.resolve(__dirname, "node_modules/happyrhino"),

                // Base directory for the application
                "base": srcPath,

                // Application name
                "name": "GitBook",

                // Mode debug
                "debug": true,

                // Main entry point for application
                "main": "main",

                // HTML entry point
                'index': grunt.file.read(path.resolve(srcPath, "index.html")),

                // Build output directory
                "build": path.resolve(__dirname, "build"),

                // Static files mappage
                "static": {
                    "ace": path.resolve(srcPath, "vendors/ace-builds/src"),
                    "mathjax": path.resolve(srcPath, "vendors/MathJax"),
                    "fonts": path.resolve(srcPath, "resources", "fonts"),
                    "fonts/octicons": path.resolve(srcPath, "vendors/octicons/octicons"),
                    "fonts/fontawesome": path.resolve(srcPath, "vendors/fontawesome/fonts"),
                    "images": path.resolve(srcPath, "resources", "images")
                },

                // Stylesheet entry point
                "style": path.resolve(srcPath, "resources/stylesheets/main.less"),

                // Modules paths
                'paths': {
                    "ace": "vendors/ace-builds/src/ace",
                    "bootstrap": "vendors/bootstrap/js"
                },
                "shim": {
                    "main": {
                        deps: [
                            'hr/dom',
                            'bootstrap/carousel',
                            'bootstrap/dropdown',
                            'bootstrap/button',
                            'bootstrap/modal',
                            'bootstrap/affix',
                            'bootstrap/alert',
                            'bootstrap/collapse',
                            'bootstrap/tooltip',
                            'bootstrap/popover',
                            'bootstrap/scrollspy',
                            'bootstrap/tab',
                            'bootstrap/transition'
                        ]
                    },
                    "ace": {
                        exports: "ace"
                    }
                },
                'args': {},
                'options': {}
            }
        },
        nodewebkit: {
            options: {
                appName: "GitBook",
                appVersion: pkg.version,
                buildDir: './appbuilds/releases',
                cacheDir: './appbuilds/cache',
                platforms: ['win', 'osx', 'linux32', 'linux64'],
                macIcns: "./build/static/images/icons/512.icns",
                macCredits: "./src/credits.html",
                winIco: "./build/static/images/icons/512.ico",
                version: NW_VERSION,
                zip: false
            },
            src: [
                "./**/*",
                "!./src/**",
                "./src/dirname.js",
                "!./appbuilds/**",
                "!./node_modules/hr.js/**",
                "!./node_modules/grunt-*/**",
                "!./node_modules/grunt/**",
                "!./node_modules/nw-gyp/**"
            ]
        },
        clean: {
            build: ['build/'],
            releases: ['appbuilds/releases/'],
            mathjax: [
                'src/vendors/MathJax/docs',
                'src/vendors/MathJax/test',
                'src/vendors/MathJax/unpacked',
                'src/vendors/MathJax/.gitignore',
                'src/vendors/MathJax/bower.json',
                'src/vendors/MathJax/localization',
                'src/vendors/MathJax/fonts',
                'src/vendors/MathJax/images',
                'src/vendors/MathJax/jax/output/NativeMML',
                'src/vendors/MathJax/jax/output/HTML-CSS',
                'src/vendors/MathJax/jax/output/SVG/fonts/Asana-Math',
                'src/vendors/MathJax/jax/output/SVG/fonts/Gyre-Pagella',
                'src/vendors/MathJax/jax/output/SVG/fonts/Gyre-Termes',
                'src/vendors/MathJax/jax/output/SVG/fonts/Latin-Modern',
                'src/vendors/MathJax/jax/output/SVG/fonts/Neo-Euler',
                'src/vendors/MathJax/jax/output/SVG/fonts/STIX-Web',
                'src/vendors/MathJax/*.md'
            ]
        },
        exec: {
            build_mac_release: {
                command: "./scripts/build_mac.sh",
                cwd: './',
                stdout: true,
                stderr: true
            },
            build_win_release: {
                command: "./scripts/build_win.sh",
                cwd: './',
                stdout: true,
                stderr: true
            },
            build_linux32_release: {
                command: "./scripts/build_linux32.sh",
                cwd: './',
                stdout: true,
                stderr: true
            },
            build_linux64_release: {
                command: "./scripts/build_linux64.sh",
                cwd: './',
                stdout: true,
                stderr: true
            }
        },
        copy: {
            // Installer for linux
            linux32Installer: {
                cwd: './',
                src: 'scripts/install_linux.sh',
                dest: './appbuilds/releases/GitBook/linux32/install.sh'
            },
            // Entry point for linux
            linux32Start: {
                cwd: './',
                src: 'scripts/linux_start.sh',
                dest: './appbuilds/releases/GitBook/linux32/start.sh'
            },
            // Icon for linux
            linux32Icon: {
                cwd: './',
                src: './build/static/images/icons/128.png',
                dest: './appbuilds/releases/GitBook/linux32/icon.png'
            },
            // Installer for linux
            linux64Installer: {
                cwd: './',
                src: 'scripts/install_linux.sh',
                dest: './appbuilds/releases/GitBook/linux64/install.sh'
            },
            // Entry point for linux
            linux64Start: {
                cwd: './',
                src: 'scripts/linux_start.sh',
                dest: './appbuilds/releases/GitBook/linux64/start.sh'
            },
            // Icon for linux
            linux64Icon: {
                cwd: './',
                src: './build/static/images/icons/128.png',
                dest: './appbuilds/releases/GitBook/linux64/icon.png'
            },
        },
        "github-release": {
            options: {
                repository: 'GitbookIO/editor',
                auth: {
                    user: process.env.GH_USERNAME,
                    password: process.env.GH_PASSWORD
                },
                release: {
                    tag_name: pkg.version,
                    name: pkg.version,
                    draft: true,
                    prerelease: false
                }
            },
            files: {
                src: [
                    "./appbuilds/releases/gitbook-linux32.tar.gz",
                    "./appbuilds/releases/gitbook-linux64.tar.gz",
                    "./appbuilds/releases/gitbook-mac.dmg",
                    "./appbuilds/releases/gitbook-win.zip"
                ],
            },
        }
    });

    // Build
    grunt.registerTask('build', [
        'bower-install-simple',
        'clean:mathjax',
        'hr:app'
    ]);

    // Release
    grunt.registerTask('build-mac', [
        'exec:build_mac_release'
    ]);
    grunt.registerTask('build-win', [
        'exec:build_win_release'
    ]);
    grunt.registerTask('build-linux32', [
        'copy:linux32Installer',
        'copy:linux32Start',
        'copy:linux32Icon',
        'exec:build_linux32_release'
    ]);
    grunt.registerTask('build-linux64', [
        'copy:linux64Installer',
        'copy:linux64Start',
        'copy:linux64Icon',
        'exec:build_linux64_release'
    ]);
    grunt.registerTask('build-apps', [
        'clean:build',
        'clean:releases',
        'build',
        'nodewebkit',
        'build-linux32',
        'build-linux64',
        'build-mac',
        'build-win'
    ]);
    grunt.registerTask('publish', [
        'build-apps',
        'github-release'
    ]);

    grunt.registerTask('default', [
        'build'
    ]);
};
