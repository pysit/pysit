#! /bin/bash
# For MAC OSx, we can only use the petsc 3.6.0.
#INSTALL PETSC4PY USING BREW
echo -n  "Check if brew is installed : "
Status=$(which brew)
if [[ $Status == "" ]]; then
    echo "brew is not install on your mac \n please go to : http://brew.sh and install it" response
    exit
else
    echo "OK"
fi

echo -n  "Check if gcc is installed : "
Status=$(which gcc-8)
if [[ $Status == "" ]]; then
	read -r -p "gcc is not install do you want to install it ? [y/N] " response
  case $response in
        [yY][eE][sS]|[yY])
    		brew install gcc --without-multilib
    		;;
        *)
    		exit
    		;;
    esac
else
	echo "OK"
fi

echo -n "Check if bison is installed : "
Status=$(which bison)
if [[ $Status == "" ]]; then
	read -r -p "bison is not install do you want to install it ? [y/N] " response
	case $response in
   	[yY][eE][sS]|[yY])
    		brew install bison
    		;;
    	*)
    		exit
    		;;
    	esac
else
	echo "OK"
fi
echo -n "Check if flex is installed : "
Status=$(which flex)
if [[ $Status == "" ]]; then
	read -r -p "flex is not install do you want to install it ? [y/N] " response
	case $response in
    	[yY][eE][sS]|[yY])
    		brew install flex
    		;;
   	 *)
    		exit
    		;;
    	esac
else
	echo "OK"
fi
echo -n "Check if valgrind is installed : "
Status=$(which valgrind)
if [[ "$Status" == "" ]]; then
	read -r -p "valgrind is not install do you want to install it ? [y/N] " response
	case $response in
    	[yY][eE][sS]|[yY])
    		brew install valgrind
    		;;
    	*)
    		exit
    		;;
    	esac
else
	echo "OK"
fi

echo -n "Check if curl is installed : "
Status=$(which curl)
if [[ "$Status" == "" ]]; then
	read -r -p "curl is not install do you want to install it ? [y/N] " response
	case $response in
    	[yY][eE][sS]|[yY])
    		brew install curl
    		;;
    	*)
    		exit
    		;;
    	esac
else
	echo "OK"
fi

echo -n "Check if wget is installed : "
Status=$(which wget)
if [[ "$Status" == "" ]]; then
	read -r -p "wget is not install do you want to install it ? [y/N] " response
	case $response in
    	[yY][eE][sS]|[yY])
    		brew install wget
    		;;
    	*)
    		exit
    		;;
    	esac
else
	echo "OK"
fi

# Now we can start the installation of petsc
while [[ true ]]; do
	read -p "Please specify an installation path or press enter to install petsc in the curent directory $(pwd)  : " path_petsc
	if [[ "$path_petsc" == "" ]]; then
		break
	elif [[ -d "$path_petsc" ]]; then
		cd $path_petsc
		break
	else
		echo "the path specified do not exit please re type it again"
	fi
done

echo "dowloading petsc :"
#export PATH=/usr/local/Cellar/gcc/*/bin/:$PATH
# # # If the user as already install xcode g++ link to clang compiller so we force to use brew gcc-5
# # # sometimes gfortran is not provided by LLVM but 5.x GNU is
curl http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.6.1.tar.gz | tar xz
cd petsc-3.6.1
if [[ $? -ne 0 ]]; then
	exit
fi
# ./configure --with-cc=gcc-8 --with-cxx=g++-8 --with-fc=gfortran-8 --download-fblaslapack \
#             --download-cmake --download-mpich --download-ptscotch --with-scalar-type=complex \
#             --download-metis --download-parmetis --download-suitesparse --download-triangle \
#             --download-superlu --download-superlu_dist --download-scalapack --download-mumps \
#             --with-shared-libraries --with-python --PETSC_ARCH=arch-osx

python2 './configure' '--with-cc=gcc-8' '--with-cxx=g++-8' '--with-fc=gfortran-8' '--download-fblaslapack' \
            '--download-cmake' '--download-mpich' '--download-ptscotch' '--with-scalar-type=complex'\
            '--download-metis' '--download-parmetis' '--download-suitesparse' '--download-triangle' \
            '--download-superlu' '--download-superlu_dist' '--download-scalapack' '--download-mumps' \
            '--with-shared-libraries' '--with-python' '--PETSC_ARCH=arch-osx'

if [[ $? -ne 0 ]]; then
	exit
fi
export PETSC_DIR=$(pwd)
export PETSC_ARCH=arch-osx
make all
if [[ $? -ne 0 ]]; then
	exit
fi
make test
if [[ $? -ne 0 ]]; then
	exit
fi
make streams NPMAX=8
if [[ $? -ne 0 ]]; then
	exit
fi
export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
export PATH=$PETSC_DIR/$PETSC_ARCH/lib:$PATH
export PATH=$PETSC_DIR/$PETSC_ARCH/include:$PATH
#we have to reinstall mpi4py it will use the lastest version of mpi installed by PETSc
pip uninstall mpi4py
pip install mpi4py
echo "Installing petsc4py"
cd ..

# You can either use the flag --download-petsc4py in the configure argument
# and add $PETSC_DIR/arch-osx/lib to PYTHONPATH
# or launch this rest of the script

# curl https://files.pythonhosted.org/packages/91/8c/2c5d593b5dc7aff46bd56b7c71fc5550bd342c8295440eb8c9cb255f2e71/petsc4py-3.9.1.tar.gz | tar xz
curl https://bitbucket.org/petsc/petsc4py/downloads/petsc4py-3.6.0.tar.gz | tar xz
wget https://bitbucket.org/petsc/petsc4py/downloads/petsc4py-3.6.0.tar.gz
tar zxf petsc4py-3.6.0.tar.gz

cd petsc4py-3.6.0
python setup.py install
cd ..
rm -r petsc4py-3.6.0
#To insure that MPI4Py uses the right lib check the mpi.cfg
#it is in : your_anaconda_dir/lib/python2.7/site-packages/mpi4py/mpi.cfg
#it should be :
#mpicc = YOUR_PETSc_DIR/arch-linux/bin/mpicc
#mpicxx = YOUR_PETSc_DIR/arch-linux/bin/mpicxx
#mpif77 = YOUR_PETSc_DIR/arch-linux/bin/mpif77
#mpif90 = YOUR_PETSc_DIR/arch-linux/bin/mpif90
