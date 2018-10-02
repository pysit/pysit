#! /bin/bash
echo -n  "Check if gcc is installed : "
Status=$(which gcc)
if [[ $Status == "" ]]; then
	read -r -p "gcc is not install do you want to install it ? [y/N] " response
	case $response in
        [yY][eE][sS]|[yY])
    		sudo apt-get install gcc
    		;;
        *)
    		exit
    		;;
    esac
else
	echo "OK"
fi
echo -n "Check if g++ is installed : "
Status=$(which g++)
if [[ $Status == "" ]]; then
	read -r -p "g++ is not install do you want to install it ? [y/N] " response
	case $response in
        [yY][eE][sS]|[yY])
    		sudo apt-get install g++
    		;;
        *)
    		exit
    		;;
    esac
else
	echo "OK"
fi

echo -n "Check if gfortran is installed : "
Status=$(which gfortran)
if [[ $Status == "" ]]; then
	read -r -p "gfortran is not install do you want to install it ? [y/N] " response
	case $response in
    	[yY][eE][sS]|[yY])
    		sudo apt-get install gfortran
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
    		sudo apt-get install bison
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
    		sudo apt-get install flex
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
    		sudo apt-get install valgrind
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
    		sudo apt-get install curl
    		;;
    	*)
    		exit
    		;;
    	esac
else
	echo "OK"
fi

echo -n "Check if pip is installed : "
Status=$(which pip)
if [[ "$Status" == "" ]]; then
echo "pip is not install you have to install it in order to finish the installation"

exit

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
curl http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.9.1.tar.gz | tar xz
cd petsc-3.9.1
if [[ $? -ne 0 ]]; then
	exit
fi
./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-cmake --download-fblaslapack --download-mpich --download-ptscotch --with-scalar-type=complex --download-metis --download-parmetis --download-suitesparse --download-triangle --download-superlu --download-superlu_dist --download-scalapack --download-mumps --with-shared-libraries --with-python --PETSC_ARCH=arch-linux
if [[ $? -ne 0 ]]; then
	exit
fi
export PETSC_DIR=$(pwd)
export PETSC_ARCH=arch-linux
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

pip uninstall mpi4py
pip install mpi4py --user
echo "Installing petsc4py"
cd ..
curl https://files.pythonhosted.org/packages/91/8c/2c5d593b5dc7aff46bd56b7c71fc5550bd342c8295440eb8c9cb255f2e71/petsc4py-3.9.1.tar.gz | tar xz
cd petsc4py-3.9.1
python setup.py install
if [[ $? -ne 0 ]]; then
	exit
fi
cd ..
rm -r petsc4py-3.9.1

#To insure that MPI4Py uses the right lib check the mpi.cfg
#it is in : your_anaconda_dir/lib/python2.7/site-packages/mpi4py/mpi.cfg
#it should be :
#mpicc = YOUR_PETSc_DIR/arch-linux/bin/mpicc
#mpicxx = YOUR_PETSc_DIR/arch-linux/bin/mpicxx
#mpif77 = YOUR_PETSc_DIR/arch-linux/bin/mpif77
#mpif90 = YOUR_PETSc_DIR/arch-linux/bin/mpif90
