/*
nvcc collider31.cu -o collider31 -lglut -lm -lGLU -lGL --use_fast_math  -O3  -Xptxas  "-warn-lmem-usage -warn-spills" -arch=sm_52
nvcc collider31.cu -o collider31 -lglut -lm -lGLU -lGL -prec-div=false -prec-sqrt=false -ftz=true -O3 
nvcc collider31.cu -o collider31nofast -lglut -lm -lGLU -lGL -O3 
*/

#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include "stdio.h"
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <signal.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

#define BLOCKSIZE 256

#define NUMBEROFEARTHRADIFORMOONMATERIAL 20.0

//Global to hold the time of the collision
double RunTime = 0.0;

//Continue and branch run globals
int TypeOfRun = 0;
char RootFolderName[256] = "";
double AddedRunTime = 0;

//Globals for files
FILE *RunStatsFile;
FILE *PosAndVelFile;
FILE *StartPosAndVelFile;
FILE *ContinueRunStatsFile;
FILE *ContinueRunPosAndVelFile;

//Globals to hold positions, velocities, and forces on both the GPU and CPU
float4 *PlaceHolder; //needs to be hard defined for cuda
float4 *Pos, *Vel, *Force;
float4 *Pos_DEV0, *Vel_DEV0, *Force_DEV0;

float4 *PosFstHalf_0, *VelFstHalf_0, *ForceFstHalf_0;
float4 *PosSndHalf_0, *VelSndHalf_0;
float4 *PosFstHalf_1, *VelFstHalf_1;
float4 *PosSndHalf_1, *VelSndHalf_1, *ForceSndHalf_1;

//Globals to setup the kernals
dim3 BlockConfig, GridConfig;
int NumberOfGpus, Gpu0Access, Gpu1Access;

//Globals to be set by the setRunParameters function
double UnitLength = -1.0;
double Diameter = -1.0;
double UnitMass = -1.0;
double MassSi = -1.0;
double MassFe = -1.0;
double MassOfBody1 = -1.0;
double MassOfBody2 = -1.0;
double UnitTime = -1.0;
double Gravity = -1.0;

int NSi = -1; 
int NSi1 = -1; 
int NSi2 = -1;
int NFe = -1; 
int NFe1 = -1; 
int NFe2 = -1;

//Globals to be set by the findEarthAndMoon function
int NumberOfEarthElements = -1;
int NumberOfMoonElements = -1;
int *EarthIndex;
int *MoonIndex;

//Global to trigger printing collision stats to the screen
int PrintCollisionStats = 0; 

//Global to trigger printing continue stats to the screen
int PrintContinueStats = 0;

//Globals for the run to be read in from the runSetup file
float3 InitialPosition1;
float3 InitialPosition2;
float3 InitialVelocity1;
float3 InitialVelocity2;
float4 InitialSpin1;
float4 InitialSpin2;
float3 BranchPosition1;
float3 BranchPosition2;
float3 BranchVelocity1;
float3 BranchVelocity2;
float4 BranchSpin1;
float4 BranchSpin2;
double FractionEarthMassOfBody1;	//Mass of body 1 as a proportion of the Earth's mass
double FractionEarthMassOfBody2;	//Mass of body 2 as a proportion of the Earth's mass
double FractionFeBody1;			//Percent by mass of iron in body 1
double FractionSiBody1;			//Percent by mass of silicate in body 1
double FractionFeBody2;			//Percent by mass of iron in body 2
double FractionSiBody2;			//Percent by mass of silicate in body 2
float DampRateBody1;
float DampRateBody2;
float EnergyTargetBody1;
float EnergyTargetBody2;
int N;
float TotalRunTime;
float BranchRunTime;
float DampTime;
float DampRestTime;
float EnergyAdjustmentTime;
float EnergyAdjustmentRestTime;
float SpinRestTime;
float BranchSpinRestTime;
float SetupTime;
float Dt;
int WriteToFile;
int RecordRate;
double DensityFe;			//Density of iron in kilograms meterE-3 (Canup science 2012)
double DensitySi;			//Density of silcate in kilograms meterE-3 (Canup science 2012)
double KFe;
double KSi;
double KRFe;
double KRSi;
double SDFe;
double SDSi;
int DrawRate;
int DrawQuality;
int UseMultipleGPU;
double UniversalGravity;	//Universal gravitational constant in kilometersE3 kilogramsE-1 and secondsE-2 (??? source)
double MassOfEarth;
double MassOfMoon;
double AngularMomentumEarthMoonSystem;
double EarthAxialTilt;
double MoonAxialTilt;
double Pi;

void readRunParameters()
{
	ifstream data;
	string name;
	if(TypeOfRun == 0)
	{
		data.open("RunSetup");
	}
	else if(TypeOfRun == 1)
	{
		data.open("RootSetup");
	}
	else if(TypeOfRun == 2)
	{
		data.open("RunSetup");
		if(data.is_open() != 1) data.open("RootSetup");
	}
	else 
	{
		printf("\nTSU Error bad TypeOfRun selected\n");
		exit(0);
	}
	
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> InitialPosition1.x;
		getline(data,name,'=');
		data >> InitialPosition1.y;
		getline(data,name,'=');
		data >> InitialPosition1.z;
		getline(data,name,'=');
		data >> InitialPosition2.x;
		getline(data,name,'=');
		data >> InitialPosition2.y;
		getline(data,name,'=');
		data >> InitialPosition2.z;
		
		getline(data,name,'=');
		data >> InitialVelocity1.x;
		getline(data,name,'=');
		data >> InitialVelocity1.y;
		getline(data,name,'=');
		data >> InitialVelocity1.z;
		getline(data,name,'=');
		data >> InitialVelocity2.x;
		getline(data,name,'=');
		data >> InitialVelocity2.y;
		getline(data,name,'=');
		data >> InitialVelocity2.z;
		
		getline(data,name,'=');
		data >> InitialSpin1.x;
		getline(data,name,'=');
		data >> InitialSpin1.y;
		getline(data,name,'=');
		data >> InitialSpin1.z;
		getline(data,name,'=');
		data >> InitialSpin1.w;
		
		getline(data,name,'=');
		data >> InitialSpin2.x;
		getline(data,name,'=');
		data >> InitialSpin2.y;
		getline(data,name,'=');
		data >> InitialSpin2.z;
		getline(data,name,'=');
		data >> InitialSpin2.w;
		
		getline(data,name,'=');
		data >> FractionEarthMassOfBody1;
		getline(data,name,'=');
		data >> FractionEarthMassOfBody2;
		
		getline(data,name,'=');
		data >> FractionFeBody1;
		getline(data,name,'=');
		data >> FractionSiBody1;
		getline(data,name,'=');
		data >> FractionFeBody2;
		getline(data,name,'=');
		data >> FractionSiBody2;
		
		getline(data,name,'=');
		data >> DampRateBody1;
		getline(data,name,'=');
		data >> DampRateBody2;
		
		getline(data,name,'=');
		data >> EnergyTargetBody1;
		getline(data,name,'=');
		data >> EnergyTargetBody2;
		
		getline(data,name,'=');
		data >> N;
		
		getline(data,name,'=');
		data >> TotalRunTime;
		getline(data,name,'=');
		data >> DampTime;
		getline(data,name,'=');
		data >> DampRestTime;
		getline(data,name,'=');
		data >> EnergyAdjustmentTime;
		getline(data,name,'=');
		data >> EnergyAdjustmentRestTime;
		getline(data,name,'=');
		data >> SpinRestTime;
		
		getline(data,name,'=');
		data >> Dt;
		
		getline(data,name,'=');
		data >> WriteToFile;
		
		getline(data,name,'=');
		data >> RecordRate;
		
		getline(data,name,'=');
		data >> DensityFe;
		getline(data,name,'=');
		data >> DensitySi;
		
		getline(data,name,'=');
		data >> KFe;
		getline(data,name,'=');
		data >> KSi;
		getline(data,name,'=');
		data >> KRFe;
		getline(data,name,'=');
		data >> KRSi;
		getline(data,name,'=');
		data >> SDFe;
		getline(data,name,'=');
		data >> SDSi;
		
		getline(data,name,'=');
		data >> DrawRate;
		getline(data,name,'=');
		data >> DrawQuality;
		
		getline(data,name,'=');
		data >> UseMultipleGPU;
		
		getline(data,name,'=');
		data >> UniversalGravity;
		getline(data,name,'=');
		data >> MassOfEarth;
		getline(data,name,'=');
		data >> MassOfMoon;
		getline(data,name,'=');
		data >> AngularMomentumEarthMoonSystem;
		getline(data,name,'=');
		data >> EarthAxialTilt;
		getline(data,name,'=');
		data >> MoonAxialTilt;
		getline(data,name,'=');
		data >> Pi;
	}
	else
	{
		printf("\nTSU Error could not open run or root Setup file\n");
		exit(0);
	}
	data.close();
}

void readBranchParameters()
{
	ifstream data;
	string name;
	data.open("BranchSetup");
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> BranchPosition1.x;
		getline(data,name,'=');
		data >> BranchPosition1.y;
		getline(data,name,'=');
		data >> BranchPosition1.z;
		getline(data,name,'=');
		data >> BranchPosition2.x;
		getline(data,name,'=');
		data >> BranchPosition2.y;
		getline(data,name,'=');
		data >> BranchPosition2.z;
		
		getline(data,name,'=');
		data >> BranchVelocity1.x;
		getline(data,name,'=');
		data >> BranchVelocity1.y;
		getline(data,name,'=');
		data >> BranchVelocity1.z;
		getline(data,name,'=');
		data >> BranchVelocity2.x;
		getline(data,name,'=');
		data >> BranchVelocity2.y;
		getline(data,name,'=');
		data >> BranchVelocity2.z;
		
		getline(data,name,'=');
		data >> BranchSpin1.x;
		getline(data,name,'=');
		data >> BranchSpin1.y;
		getline(data,name,'=');
		data >> BranchSpin1.z;
		getline(data,name,'=');
		data >> BranchSpin1.w;
		
		getline(data,name,'=');
		data >> BranchSpin2.x;
		getline(data,name,'=');
		data >> BranchSpin2.y;
		getline(data,name,'=');
		data >> BranchSpin2.z;
		getline(data,name,'=');
		data >> BranchSpin2.w;
		
		getline(data,name,'=');
		data >> BranchSpinRestTime;
		getline(data,name,'=');
		data >> BranchRunTime;
	}
	else
	{
		printf("\nTSU Error could not open Branch Setup file\n");
		exit(0);
	}
	data.close();
}

void setRunParameters()
{
	double massBody1 = MassOfEarth*FractionEarthMassOfBody1;
	double massBody2 = MassOfEarth*FractionEarthMassOfBody2;
	if(FractionFeBody1 + FractionSiBody1 != 1.0) 
	{
		printf("\nTSU Error: body1 fraction don't add to 1\n");
		exit(0);
	}
	if(FractionFeBody2 + FractionSiBody2 != 1.0) 
	{
		printf("\nTSU Error: body2 fraction don't add to 1\n");
		exit(0);
	}
	double totalMassOfFeBody1 = FractionFeBody1*massBody1;
	double totalMassOfSiBody1 = FractionSiBody1*massBody1;
	double totalMassOfFeBody2 = FractionFeBody2*massBody2;
	double totalMassOfSiBody2 = FractionSiBody2*massBody2;
	double totalMassOfFe = totalMassOfFeBody1 + totalMassOfFeBody2;
	double totalMassOfSi = totalMassOfSiBody1 + totalMassOfSiBody2;
	double massFe;
	double massSi;
	double diameterOfElement;
	
	if(totalMassOfFe != 0.0) NFe = (double)N*(DensitySi/DensityFe)/(totalMassOfSi/totalMassOfFe + DensitySi/DensityFe);
	else NFe = 0;
	NSi = N - NFe;
	
	if(totalMassOfFe != 0.0) NFe1 = NFe*totalMassOfFeBody1/totalMassOfFe; 
	else NFe1 = 0;
	
	NFe2 = NFe - NFe1;
	
	if(totalMassOfSi != 0.0) NSi1 = NSi*totalMassOfSiBody1/totalMassOfSi; 
	else NSi1 = 0;
	
	NSi2 = NSi - NSi1;
	
	if(NFe != 0) massFe = totalMassOfFe/NFe;
	else massFe = 0.0;
	if(NSi != 0) massSi = totalMassOfSi/NSi;
	else massSi = 0.0;
	
	if(NSi != 0) diameterOfElement = pow((6.0*massSi)/(Pi*DensitySi), (1.0/3.0));
	else diameterOfElement = pow((6.0*massFe)/(Pi*DensityFe), (1.0/3.0));
	
	UnitLength = diameterOfElement;
	
	if(NSi != 0) UnitMass = massSi;
	else UnitMass = massFe;
	
	if(NSi != 0) UnitTime = sqrt((6.0*massSi*(double)NSi)/(UniversalGravity*Pi*DensitySi*totalMassOfSi));
	else if(NFe != 0) UnitTime = sqrt((6.0*massFe*(double)NFe)/(UniversalGravity*Pi*DensityFe*totalMassOfFe));
	else 
	{
		printf("TSU Error: No mass, function setRunParameters\n");
		exit(0);
	}
	
	//In this system this is what sets the length unit, the time unit, and the mass unit. 
	Diameter = 1.0;
	Gravity = 1.0;

	if(NSi != 0)
	{
		MassSi = 1.0;
		MassFe = DensityFe/DensitySi;
	}
	else if(NFe != 0)
	{
		MassFe = 1.0;
	}
	else 
	{
		printf("TSU Error: No mass, function setRunParameters\n");
		exit(0);
	}
	
	//Setting mass of bodies in our units
	MassOfBody1 = massBody1/UnitMass;
	MassOfBody2 = massBody2/UnitMass;
	
	//Putting Initial positions into our units
	InitialPosition1.x /= UnitLength;
	InitialPosition1.y /= UnitLength;
	InitialPosition1.z /= UnitLength;

	InitialPosition2.x /= UnitLength;
	InitialPosition2.y /= UnitLength;
	InitialPosition2.z /= UnitLength;

	//Putting Initial Velocities into our units
	InitialVelocity1.x *= UnitTime/UnitLength;
	InitialVelocity1.y *= UnitTime/UnitLength;
	InitialVelocity1.z *= UnitTime/UnitLength;

	InitialVelocity2.x *= UnitTime/UnitLength;
	InitialVelocity2.y *= UnitTime/UnitLength;
	InitialVelocity2.z *= UnitTime/UnitLength;

	//Putting Initial Angule Velocities into our units
	InitialSpin1.w *= UnitTime/3600.0;

	InitialSpin2.w *= UnitTime/3600.0;
	
	//Putting Run times into our units
	TotalRunTime *= 3600.0/UnitTime;
	DampTime *= 3600.0/UnitTime;
	DampRestTime *= 3600.0/UnitTime;
	EnergyAdjustmentTime *= 3600.0/UnitTime;
	EnergyAdjustmentRestTime *= 3600.0/UnitTime;
	SpinRestTime *= 3600.0/UnitTime;
	SetupTime = (DampTime + DampRestTime + EnergyAdjustmentTime + EnergyAdjustmentRestTime + SpinRestTime); 
	
	KFe *= UnitTime*UnitTime*UnitLength/UnitMass;
	KSi *= UnitTime*UnitTime*UnitLength/UnitMass;
}

void setBranchParameters()
{
	//Putting Branch positions into our units
	BranchPosition1.x /= UnitLength;
	BranchPosition1.y /= UnitLength;
	BranchPosition1.z /= UnitLength;

	BranchPosition2.x /= UnitLength;
	BranchPosition2.y /= UnitLength;
	BranchPosition2.z /= UnitLength;

	//Putting Branch Velocities into our units
	BranchVelocity1.x *= UnitTime/UnitLength;
	BranchVelocity1.y *= UnitTime/UnitLength;
	BranchVelocity1.z *= UnitTime/UnitLength;

	BranchVelocity2.x *= UnitTime/UnitLength;
	BranchVelocity2.y *= UnitTime/UnitLength;
	BranchVelocity2.z *= UnitTime/UnitLength;

	//Putting Branch Angule Velocities into our units
	BranchSpin1.w *= UnitTime/3600.0;

	BranchSpin2.w *= UnitTime/3600.0;
	
	//Putting Branch Run times into our units
	BranchSpinRestTime *= 3600.0/UnitTime;
	BranchRunTime *= 3600.0/UnitTime;
}

//Globals for setting up the viewing window 
int XWindowSize = 2500;
int YWindowSize = 2500; 
double Near = 0.2;
double Far = 600.0;

double ViewBoxSize = 300.0;

GLdouble Left = -ViewBoxSize;
GLdouble Right = ViewBoxSize;
GLdouble Bottom = -ViewBoxSize;
GLdouble Top = ViewBoxSize;
GLdouble Front = ViewBoxSize;
GLdouble Back = -ViewBoxSize;

//Direction here your eye is located location

double EyeX = 100.0;
double EyeY = 100.0;
double EyeZ = 100.0;

//Where you are looking

double CenterX = 0.0;
double CenterY = 0.0;
double CenterZ = 0.0;

//Up vector for viewing

double UpX = 0.0;
double UpY = 1.0;
double UpZ = 0.0;

void createFolderForNewRun()
{   	
	//Create output folder to store run parameters and run positions and velocities
	time_t t = time(0); 
	struct tm * now = localtime( & t );
	int month = now->tm_mon + 1, day = now->tm_mday, curTimeHour = now->tm_hour, curTimeMin = now->tm_min;
	stringstream smonth, sday, stimeHour, stimeMin;
	smonth << month;
	sday << day;
	stimeHour << curTimeHour;
	stimeMin << curTimeMin;
	string monthday;
	if (curTimeMin <= 9)	monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":0" + stimeMin.str();
	else			monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":" + stimeMin.str();
	string foldernametemp = "Run:" + monthday;
	const char *foldername = foldernametemp.c_str();
	mkdir(foldername , S_IRWXU|S_IRWXG|S_IRWXO);
	chdir(foldername);
	
	//Copying the RunSetup file into the run folder
	FILE *runSetupIn;
	FILE *runSetupOut;
	long sizeOfFile;
  	char * buffer;
    	
    	runSetupIn = fopen("../RunSetup", "rb");
    	fseek (runSetupIn , 0 , SEEK_END);
  	sizeOfFile = ftell (runSetupIn);
  	rewind (runSetupIn);
  	buffer = (char*) malloc (sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, runSetupIn);
  	
    	runSetupOut = fopen("RunSetup", "wb");
    	fwrite (buffer, 1, sizeOfFile, runSetupOut);

	fclose(runSetupIn);
	fclose(runSetupOut);
	free (buffer);
}

void createFolderForBranchRun(const char* rootFolder)
{   	
	//Create output folder to store run parameters and run positions and velocities
	time_t t = time(0); 
	struct tm * now = localtime( & t );
	int month = now->tm_mon + 1, day = now->tm_mday, curTimeHour = now->tm_hour, curTimeMin = now->tm_min;
	stringstream smonth, sday, stimeHour, stimeMin;
	smonth << month;
	sday << day;
	stimeHour << curTimeHour;
	stimeMin << curTimeMin;
	string monthday;
	if (curTimeMin <= 9)	monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":0" + stimeMin.str();
	else			monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":" + stimeMin.str();
	string foldernametemp = "BranchRun:" + monthday;
	const char *foldername = foldernametemp.c_str();
	mkdir(foldername , S_IRWXU|S_IRWXG|S_IRWXO);
	chdir(foldername);
	
	FILE *fileIn;
	FILE *fileOut;
	long sizeOfFile;
  	char * buffer;
  	char path[256];
  	
  	//Copying the RunSetup file into the branch run folder
  	strcpy(path,  "../");
  	strcat(path, rootFolder);
  	strcat(path,"/RunSetup");
    	
    	fileIn = fopen(path, "rb");
    	if(fileIn == NULL)
    	{
    		printf("\n\n The RunSetup file does not exist\n\n");
    		exit(0);
    	}
    	
    	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell (fileIn);
  	rewind (fileIn);
  	buffer = (char*) malloc (sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
  	
    	fileOut = fopen("RootSetup", "wb");
    	fwrite (buffer, 1, sizeOfFile, fileOut);
    	fclose(fileOut);
    	
    	fileOut = fopen("RunSetup", "wb");
    	fwrite (buffer, 1, sizeOfFile, fileOut);

	fclose(fileIn);
	fclose(fileOut);
	free (buffer);
	
	//Copying the RunStatsFile file into the branch run folder
  	strcpy(path,  "../");
  	strcat(path, rootFolder);
  	strcat(path,"/RunStats");
    	
    	fileIn = fopen(path, "rb");
    	if(fileIn == NULL)
    	{
    		printf("\n\n The RunStats file does not exist\n\n");
    		exit(0);
    	}
    	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell (fileIn);
  	rewind (fileIn);
  	buffer = (char*) malloc (sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
  	
    	fileOut = fopen("RootRunStats", "wb");
    	fwrite (buffer, 1, sizeOfFile, fileOut);

	fclose(fileIn);
	fclose(fileOut);
	free (buffer);
	
	//Copying the Branch Positions and Velocities file into the branch run folder
  	strcpy(path,  "../");
  	strcat(path, rootFolder);
  	strcat(path,"/StartPosAndVel");
    	
    	fileIn = fopen(path, "rb");
    	if(fileIn == NULL)
    	{
    		printf("\n\n The StartPosAndVel file does not exist\n\n");
    		exit(0);
    	}
    	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell (fileIn);
  	rewind (fileIn);
  	buffer = (char*) malloc (sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
  	
    	fileOut = fopen("RootStartPosAndVel", "wb");
    	fwrite (buffer, 1, sizeOfFile, fileOut);

	fclose(fileIn);
	fclose(fileOut);
	free (buffer);
	
	//Copying the Branch setup file into the branch run folder
  	strcpy(path,  "../");
  	strcat(path,"BranchSetup");
    	
    	fileIn = fopen(path, "rb");
    	if(fileIn == NULL)
    	{
    		printf("\n\n The BranchSetup file does not exist\n\n");
    		exit(0);
    	}
    	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell (fileIn);
  	rewind (fileIn);
  	buffer = (char*) malloc (sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
  	
    	fileOut = fopen("BranchSetup", "wb");
    	fwrite (buffer, 1, sizeOfFile, fileOut);

	fclose(fileIn);
	fclose(fileOut);
	free (buffer);
}

void openNewRunFiles()
{
	RunStatsFile = fopen("RunStats", "wb");	
	PosAndVelFile = fopen("PosAndVel", "wb");
	StartPosAndVelFile = fopen("StartPosAndVel", "wb");
	ContinueRunStatsFile = fopen("ContinueRunStats", "wb");
	ContinueRunPosAndVelFile = fopen("ContinueRunPosAndVel", "wb");
}

void openBranchRunFiles()
{
	RunStatsFile = fopen("RunStats", "wb");	
	PosAndVelFile = fopen("PosAndVel", "wb");
	StartPosAndVelFile = fopen("StartPosAndVel", "wb");
	ContinueRunStatsFile = fopen("ContinueRunStats", "wb");
	ContinueRunPosAndVelFile = fopen("ContinueRunPosAndVel", "wb");
}

void openContinueRunFiles()
{
	RunStatsFile = fopen("RunStats", "wb");	
	PosAndVelFile = fopen("PosAndVel", "ab");
	//fseek(PosAndVelFile,0,SEEK_END);
	ContinueRunStatsFile = fopen("ContinueRunStats", "wb");
	ContinueRunPosAndVelFile = fopen("ContinueRunPosAndVel", "wb");
}

void recordSetupStats()
{
	float mag;
	fprintf(RunStatsFile, "The conversion parameters to take you to and from our units to the real world units follow\n");
	
	fprintf(RunStatsFile, "\nOur length unit is this many kilometers: 	UnitLength = %f", UnitLength);
	fprintf(RunStatsFile, "\nOur mass unit is this many kilograms: 		UnitMass = %e", UnitMass);
	fprintf(RunStatsFile, "\nOur time unit is this many seconds: 		UnitTime = %f\n", UnitTime);
	
	fprintf(RunStatsFile,   "\nThe initail statistics for this run in our units follow\n");
	fprintf(RunStatsFile, "\nDiameter of an element: 		Diameter = %f", Diameter);
	fprintf(RunStatsFile, "\nGravity in our units: 			Gravity = %f", Gravity);
	fprintf(RunStatsFile, "\nThe mass of a silicate element: 	MassSi = %f", MassSi);
	fprintf(RunStatsFile, "\nThe mass of an iron element: 		MassFe = %f\n", MassFe);
	
	fprintf(RunStatsFile, "\nThe push back strength of iron: 	KFe = %f", KFe);
	fprintf(RunStatsFile, "\nThe push back strength of silicate: 	KSi = %f\n", KSi);
	
	fprintf(RunStatsFile, "\nThe mass of body one: 	MassOfBody1 = %f", MassOfBody1);
	fprintf(RunStatsFile, "\nThe mass of body two: 	MassOfBody2 = %f\n", MassOfBody2);
	
	fprintf(RunStatsFile, "\nThe initial position of body one: (%f, %f, %f)", InitialPosition1.x, InitialPosition1.y, InitialPosition1.z);
	fprintf(RunStatsFile, "\nThe initial position of body two: (%f, %f, %f)\n", InitialPosition2.x, InitialPosition2.y, InitialPosition2.z);
	
	fprintf(RunStatsFile, "\nThe initial velocity of body one: (%f, %f, %f)", InitialVelocity1.x, InitialVelocity1.y, InitialVelocity1.z);
	fprintf(RunStatsFile, "\nThe initial velocity of body two: (%f, %f, %f)\n", InitialVelocity2.x, InitialVelocity2.y, InitialVelocity2.z);
	
	mag = sqrt(InitialSpin1.x*InitialSpin1.x + InitialSpin1.y*InitialSpin1.y + InitialSpin1.z*InitialSpin1.z);
	fprintf(RunStatsFile, "\nThe initial spin in revolutions per time unit of body one: (%f, %f, %f, %f)", InitialSpin1.x/mag, InitialSpin1.y/mag, InitialSpin1.z/mag, InitialSpin1.w);
	mag = sqrt(InitialSpin2.x*InitialSpin2.x + InitialSpin2.y*InitialSpin2.y + InitialSpin2.z*InitialSpin2.z);
	fprintf(RunStatsFile, "\nThe initial spin in revolutions per time unit of body two: (%f, %f, %f, %f)\n", InitialSpin2.x/mag, InitialSpin2.y/mag, InitialSpin2.z/mag, InitialSpin2.w);
	
	
	fprintf(RunStatsFile, "\nTotal number of elements: 				N = %d", N);
	fprintf(RunStatsFile, "\nTotal number of iron elements: 				NFe = %d", NFe);
	fprintf(RunStatsFile, "\nTotal number of silicate elements: 			NSi = %d", NSi);
	fprintf(RunStatsFile, "\nTotal number of iron elements in body1: 		NFe1 = %d", NFe1);
	fprintf(RunStatsFile, "\nTotal number of silicate elements in body1: 		NSi1 = %d", NSi1);
	fprintf(RunStatsFile, "\nTotal number of iron elements in body2 			NFe2: = %d", NFe2);
	fprintf(RunStatsFile, "\nTotal number of silicate elements in body2: 		NSi2 = %d\n", NSi2);
	
	fprintf(RunStatsFile, "\nTime step in our units: 	Dt = %f", Dt);
	fprintf(RunStatsFile, "\nRecord rate: 			RecordRate = %d", RecordRate);
	fprintf(RunStatsFile, "\nTotal run time in our units: 	TotalRunTime = %f\n", TotalRunTime);
	
	fprintf(RunStatsFile, "\nDamp time in our units: 			DampTime = %f", DampTime);
	fprintf(RunStatsFile, "\nDamp rest time in our units: 			DampRestTime = %f", DampRestTime);
	fprintf(RunStatsFile, "\nEnergy adjustment time in our units: 		EnergyAdjustmentTime = %f", EnergyAdjustmentTime);
	fprintf(RunStatsFile, "\nEnergy adjustment rest time in our units: 	EnergyAdjustmentRestTime = %f", EnergyAdjustmentRestTime);
	fprintf(RunStatsFile, "\nSpin rest time in our units: 			SpinRestTime = %f", SpinRestTime);
	fprintf(RunStatsFile, "\nTotal setup time in our units: 			SetupTime = %f\n", SetupTime);
}

//Creating structures to hold constants needed in the kernals
struct forceSeperateKernalConstantsStruct
{
	float GMassFeFe;
	float GMassFeSi;    
	float KFeFe;
	float KSiSi;
	float KFeSi;
	float KRFeFe;
	float KRSiSi;
	float KRFeSi;
	float KRMix;
	float ShellBreakFe;
	float ShellBreakSi;
	float ShellBreakFeSi1;
	float ShellBreakFeSi2; 
	int boarder1; 
	int boarder2;
	int boarder3;  
};

struct forceCollisionKernalConstantsStruct
{
	float GMassFeFe;
	float GMassFeSi;    
	float KFeFe;
	float KSiSi;
	float KFeSi;
	float KRFeFe;
	float KRSiSi;
	float KRFeSi;
	float KRMix;
	float ShellBreakFe;
	float ShellBreakSi;
	float ShellBreakFeSi1;
	float ShellBreakFeSi2; 
	int NFe;   
};

struct moveSeperateKernalConstantsStruct
{
	float Dt;
	float DtOverMassFe;
	float DtOverMassSi;
	int boarder1; 
	int boarder2;
	int boarder3;
};

struct moveCollisionKernalConstantsStruct
{
	float Dt;
	float DtOverMassFe;
	float DtOverMassSi;
	int NFe;
};

//Globals to hold kernal constants
forceSeperateKernalConstantsStruct ForceSeperateConstant;
forceCollisionKernalConstantsStruct ForceCollisionConstant;
moveSeperateKernalConstantsStruct MoveSeperateConstant;
moveCollisionKernalConstantsStruct MoveCollisionConstant; 

void loadKernalConstantStructures()
{
	//Force kernal seperate
	ForceSeperateConstant.GMassFeFe = Gravity*MassFe*MassFe;
	ForceSeperateConstant.GMassFeSi = Gravity*MassFe*MassSi;
	
	ForceSeperateConstant.KFeFe = 2.0*KFe;
	ForceSeperateConstant.KSiSi = 2.0*KSi;
	ForceSeperateConstant.KFeSi = KFe + KSi;
	
	ForceSeperateConstant.KRFeFe = 2.0*KFe*KRFe;
	ForceSeperateConstant.KRSiSi = 2.0*KSi*KRSi;
	ForceSeperateConstant.KRFeSi = KFe*KRFe + KSi*KRSi;
	
	if(SDFe >= SDSi) 	ForceSeperateConstant.KRMix = KFe + KSi*KRSi; 
	else 			ForceSeperateConstant.KRMix = KFe*KRFe + KSi;
	
	ForceSeperateConstant.ShellBreakFe = Diameter - Diameter*SDFe;
	ForceSeperateConstant.ShellBreakSi = Diameter - Diameter*SDSi;
	if(SDFe >= SDSi)
	{
		ForceSeperateConstant.ShellBreakFeSi1 = Diameter - Diameter*SDSi;
		ForceSeperateConstant.ShellBreakFeSi2 = Diameter - Diameter*SDFe;
	} 
	else 
	{
		ForceSeperateConstant.ShellBreakFeSi1 = Diameter - Diameter*SDFe;
		ForceSeperateConstant.ShellBreakFeSi2 = Diameter - Diameter*SDSi;
	}
	
	ForceSeperateConstant.boarder1 = NFe1;
	ForceSeperateConstant.boarder2 = NFe1 + NSi1;
	ForceSeperateConstant.boarder3 = NFe1 + NSi1 + NFe2;
	
	//Force kernal Earth Moon System
	ForceCollisionConstant.GMassFeFe = Gravity*MassFe*MassFe;
	ForceCollisionConstant.GMassFeSi = Gravity*MassFe*MassSi;
	
	ForceCollisionConstant.KFeFe = 2.0*KFe;
	ForceCollisionConstant.KSiSi = 2.0*KSi;
	ForceCollisionConstant.KFeSi = KFe + KSi;
	
	ForceCollisionConstant.KRFeFe = 2.0*KFe*KRFe;
	ForceCollisionConstant.KRSiSi = 2.0*KSi*KRSi;
	ForceCollisionConstant.KRFeSi = KFe*KRFe + KSi*KRSi;
	
	if(SDFe >= SDSi) 	ForceCollisionConstant.KRMix = KFe + KSi*KRSi; 
	else 			ForceCollisionConstant.KRMix = KFe*KRFe + KSi;
	
	ForceCollisionConstant.ShellBreakFe = Diameter - Diameter*SDFe;
	ForceCollisionConstant.ShellBreakSi = Diameter - Diameter*SDSi;
	if(SDFe >= SDSi)
	{
		ForceCollisionConstant.ShellBreakFeSi1 = Diameter - Diameter*SDSi;
		ForceCollisionConstant.ShellBreakFeSi2 = Diameter - Diameter*SDFe;
	} 
	else 
	{
		ForceCollisionConstant.ShellBreakFeSi1 = Diameter - Diameter*SDFe;
		ForceCollisionConstant.ShellBreakFeSi2 = Diameter - Diameter*SDSi;
	}
	
	ForceCollisionConstant.NFe = NFe;
	
	//Move kernal seperate	
	MoveSeperateConstant.Dt = Dt;
	MoveSeperateConstant.DtOverMassFe = Dt/MassFe;
	MoveSeperateConstant.DtOverMassSi = Dt/MassSi;
	MoveSeperateConstant.boarder1 = NFe1;
	MoveSeperateConstant.boarder2 = NSi1 + NFe1;
	MoveSeperateConstant.boarder3 = NFe1 + NSi1 + NFe2;
	
	//Move kernal Earth Moon System
	MoveCollisionConstant.Dt = Dt;
	MoveCollisionConstant.DtOverMassSi = Dt/MassSi;
	MoveCollisionConstant.DtOverMassFe = Dt/MassFe;
	MoveCollisionConstant.NFe = NFe;
}

void errorCheck(const char *message)
{
  cudaError_t  error;
  error = cudaGetLastError();

  if(error != cudaSuccess)
  {
    printf("\n CUDA ERROR: %s = %s\n", message, cudaGetErrorString(error));
    exit(0);
  }
}

void allocateCPUMemory()
{
	PlaceHolder = (float4*)malloc(N*sizeof(float4));
	Pos = (float4*)malloc(N*sizeof(float4));
	Vel = (float4*)malloc(N*sizeof(float4));
	Force = (float4*)malloc(N*sizeof(float4));
}

void checkSetupForErrors()
{
	if(N%BLOCKSIZE != 0)
	{
		printf("\nTSU Error: Number of Particles is not a multiple of the block size \n\n");
		exit(0);
	}
}

void deviceSetupSeperate()
{
	BlockConfig.x = BLOCKSIZE;
	BlockConfig.y = 1;
	BlockConfig.z = 1;
	
	GridConfig.x = (N-1)/BlockConfig.x + 1;
	GridConfig.y = 1;
	GridConfig.z = 1;
	
	cudaMalloc((void**)&Pos_DEV0, N *sizeof(float4));
	errorCheck("cudaMalloc Pos");
	cudaMalloc((void**)&Vel_DEV0, N *sizeof(float4));
	errorCheck("cudaMalloc Vel");
	cudaMalloc((void**)&Force_DEV0, N *sizeof(float4));
	errorCheck("cudaMalloc Force");
}

void deviceSetupCollision()
{
	cudaGetDeviceCount(&NumberOfGpus);
	printf("\n***** You have %d GPUs available\n", NumberOfGpus);
	errorCheck("cudaGetDeviceCount");
	cudaDeviceCanAccessPeer(&Gpu0Access,0,1);
	errorCheck("cudaDeviceCanAccessPeer0");
	cudaDeviceCanAccessPeer(&Gpu1Access,1,0);
	errorCheck("cudaDeviceCanAccessPeer1");
	if(1 < NumberOfGpus && UseMultipleGPU == 1)
	{
		printf("\n***** You will be using %d GPUs\n", NumberOfGpus);
		if(Gpu0Access == 0)
		{
			printf("\nTSU Error: Device0 can not do peer to peer\n");
		}
	
		if(Gpu1Access == 0)
		{
			printf("\nTSU Error: Device1 can not do peer to peer\n");
		}
		cudaDeviceEnablePeerAccess(1,0);
		errorCheck("cudaDeviceEnablePeerAccess");
		
		BlockConfig.x = BLOCKSIZE;
		BlockConfig.y = 1;
		BlockConfig.z = 1;
		
		GridConfig.x = ((N/2)-1)/BlockConfig.x + 1;
		GridConfig.y = 1;
		GridConfig.z = 1;
		
		cudaSetDevice(0);
		errorCheck("cudaSetDevice0");
		cudaMalloc( (void**)&PosFstHalf_0, (N/2)*sizeof(float4) );
		errorCheck("cudaMalloc PFH0");
		cudaMalloc( (void**)&PosSndHalf_0, (N/2)*sizeof(float4) );
		errorCheck("cudaMalloc PSH0");
		cudaMalloc( (void**)&VelFstHalf_0, (N/2)*sizeof(float4) );
		errorCheck("cudaMalloc VFH0");
		cudaMalloc( (void**)&VelSndHalf_0, (N/2)*sizeof(float4) );
		errorCheck("cudaMalloc VSH0");
		cudaMalloc( (void**)&ForceFstHalf_0, (N/2)*sizeof(float4) );
		errorCheck("cudaMalloc FFH0");

		cudaSetDevice(1);
		errorCheck("cudaSetDevice1");
		cudaMalloc( (void**)&PosFstHalf_1, (N/2)*sizeof(float4) );
		errorCheck("cudaMalloc PFH1");
		cudaMalloc( (void**)&PosSndHalf_1, (N/2)*sizeof(float4) );
		errorCheck("cudaMalloc PSH1");
		cudaMalloc( (void**)&VelFstHalf_1, (N/2)*sizeof(float4) );
		errorCheck("cudaMalloc VFH1");
		cudaMalloc( (void**)&VelSndHalf_1, (N/2)*sizeof(float4) );
		errorCheck("cudaMalloc VSH1");
		cudaMalloc( (void**)&ForceSndHalf_1, (N/2)*sizeof(float4) );
		errorCheck("cudaMalloc FSH1");
	}
	else
	{
		BlockConfig.x = BLOCKSIZE;
		BlockConfig.y = 1;
		BlockConfig.z = 1;
	
		GridConfig.x = (N-1)/BlockConfig.x + 1;
		GridConfig.y = 1;
		GridConfig.z = 1;
	
		cudaMalloc((void**)&Pos_DEV0, N *sizeof(float4));
		errorCheck("cudaMalloc P0");
		cudaMalloc((void**)&Vel_DEV0, N *sizeof(float4));
		errorCheck("cudaMalloc V0");
		cudaMalloc((void**)&Force_DEV0, N *sizeof(float4));
		errorCheck("cudaMalloc F0");
	}
}

void cleanUpSeperate()
{
	cudaFree(Pos_DEV0);
	cudaFree(Vel_DEV0);
	cudaFree(Force_DEV0);
	fclose(StartPosAndVelFile);
}

void cleanUpCollision()
{
	fclose(RunStatsFile);
	fclose(PosAndVelFile);
	fclose(ContinueRunStatsFile);
	fclose(ContinueRunPosAndVelFile);
	
	if(1 < NumberOfGpus && UseMultipleGPU == 1)
	{
		cudaSetDevice(0);
		errorCheck("cudaSetDevice 0");
		cudaFree(PosFstHalf_0);
		cudaFree(VelFstHalf_0);
		cudaFree(ForceFstHalf_0);
		cudaFree(PosSndHalf_0);
		cudaFree(VelSndHalf_0);
		
		cudaSetDevice(1);
		errorCheck("cudaSetDevice 0");
		cudaFree(PosFstHalf_1);
		cudaFree(VelFstHalf_1);
		cudaFree(ForceSndHalf_1);
		cudaFree(PosSndHalf_1);
		cudaFree(VelSndHalf_1);
	}
	else
	{
		cudaFree(Pos_DEV0);
		cudaFree(Vel_DEV0);
		cudaFree(Force_DEV0);
	}
}

void createBodies()
{
	float radius1, radius2, stretch;
	float volume, mag, radius, seperation;
	int test, repeatCount;
	time_t t;
	
	printf("\nCreating the raw bodies\n");
	//Creating body one
	//This assumes a 68% packing ratio of a shpere with shperes and then stretches it by strecth 
	//to safely fit all the balls in.
	stretch = 2.0;
	volume = ((4.0/3.0)*Pi*pow(Diameter,3)*(float)NFe1/0.68)*stretch;
	radius1 = pow(volume/((4.0/3.0)*Pi),(1.0/3.0));
	volume = ((4.0/3.0)*Pi*pow(Diameter,3)*(float)(NFe1 + NSi1)/0.68)*stretch;
	radius2 = pow(volume/((4.0/3.0)*Pi),(1.0/3.0));
	srand((unsigned) time(&t));
	
	repeatCount = 0;
	for(int i=0; i<NFe1; i++)
	{
		test = 0;
		while(test == 0)
		{
			Pos[i].x = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			Pos[i].y = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			Pos[i].z = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			mag = sqrt(Pos[i].x*Pos[i].x + Pos[i].y*Pos[i].y + Pos[i].z*Pos[i].z);
			radius = ((float)rand()/(float)RAND_MAX)*radius1;
			Pos[i].x *= radius/mag;
			Pos[i].y *= radius/mag;
			Pos[i].z *= radius/mag;
			test = 1;
			for(int j = 0; j < i; j++)
			{
				seperation = mag = sqrt((Pos[i].x-Pos[j].x)*(Pos[i].x-Pos[j].x) + (Pos[i].y-Pos[j].y)*(Pos[i].y-Pos[j].y) + (Pos[i].z-Pos[j].z)*(Pos[i].z-Pos[j].z));
				if(seperation < Diameter)
				{
					test = 0;
					repeatCount++;
					break;
				}
			}
		}
		Pos[i].w = 0.0;
		
		Vel[i].x = 0.0;
		Vel[i].y = 0.0;
		Vel[i].z = 0.0;
		Vel[i].w = MassFe;
	}
	
	for(int i = NFe1; i < (NFe1 + NSi1); i++)
	{
		test = 0;
		while(test == 0)
		{
			Pos[i].x = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			Pos[i].y = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			Pos[i].z = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			mag = sqrt(Pos[i].x*Pos[i].x + Pos[i].y*Pos[i].y + Pos[i].z*Pos[i].z);
			radius = ((float)rand()/(float)RAND_MAX)*(radius2-radius1) + radius1 + Diameter;
			Pos[i].x *= radius/mag;
			Pos[i].y *= radius/mag;
			Pos[i].z *= radius/mag;
			test = 1;
			for(int j = NFe1; j < i; j++)
			{
				seperation = mag = sqrt((Pos[i].x-Pos[j].x)*(Pos[i].x-Pos[j].x) + (Pos[i].y-Pos[j].y)*(Pos[i].y-Pos[j].y) + (Pos[i].z-Pos[j].z)*(Pos[i].z-Pos[j].z));
				if(seperation < Diameter)
				{
					test = 0;
					repeatCount++;
					break;
				}
			}
		}
		Pos[i].w = 1.0;
		
		Vel[i].x = 0.0;
		Vel[i].y = 0.0;
		Vel[i].z = 0.0;
		Vel[i].w = MassSi;
	}
	printf("\nrepeat count body one= %d", repeatCount);
	
	//Setting the body one's center of mass location
	for(int i=0; i<(NFe1 + NSi1); i++)
	{
		Pos[i].x += InitialPosition1.x;
		Pos[i].y += InitialPosition1.y;
		Pos[i].z += InitialPosition1.z;
	}
	
	//Creating body two
	//This assumes a 68% packing ratio of a shpere with shperes and then stretches it by strecth 
 	//to safely fit all the balls in.
	stretch = 2.0;
	volume = ((4.0/3.0)*Pi*pow(Diameter,3)*(float)NFe2/0.68)*stretch;
	radius1 = pow(volume/((4.0/3.0)*Pi),(1.0/3.0));
	volume = ((4.0/3.0)*Pi*pow(Diameter,3)*(float)(NFe2 + NSi2)/0.68)*stretch;
	radius2 = pow(volume/((4.0/3.0)*Pi),(1.0/3.0));
	srand((unsigned) time(&t));
	
	repeatCount = 0;
	for(int i = (NFe1 + NSi1); i < (NFe1 + NSi1 + NFe2); i++)
	{
		test = 0;
		while(test == 0)
		{
			Pos[i].x = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			Pos[i].y = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			Pos[i].z = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			mag = sqrt(Pos[i].x*Pos[i].x + Pos[i].y*Pos[i].y + Pos[i].z*Pos[i].z);
			radius = ((float)rand()/(float)RAND_MAX)*radius1;
			Pos[i].x *= radius/mag;
			Pos[i].y *= radius/mag;
			Pos[i].z *= radius/mag;
			test = 1;
			for(int j = (NFe1 + NSi1); j < i; j++)
			{
				seperation = mag = sqrt((Pos[i].x-Pos[j].x)*(Pos[i].x-Pos[j].x) + (Pos[i].y-Pos[j].y)*(Pos[i].y-Pos[j].y) + (Pos[i].z-Pos[j].z)*(Pos[i].z-Pos[j].z));
				if(seperation < Diameter)
				{
					test = 0;
					repeatCount++;
					break;
				}
			}
		}
		Pos[i].w = 2.0;
		
		Vel[i].x = 0.0;
		Vel[i].y = 0.0;
		Vel[i].z = 0.0;
		Vel[i].w = MassFe;
	}
	for(int i = (NFe1 + NSi1 + NFe2); i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			Pos[i].x = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			Pos[i].y = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			Pos[i].z = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			mag = sqrt(Pos[i].x*Pos[i].x + Pos[i].y*Pos[i].y + Pos[i].z*Pos[i].z);
			radius = ((float)rand()/(float)RAND_MAX)*(radius2-radius1) + radius1 + Diameter;
			Pos[i].x *= radius/mag;
			Pos[i].y *= radius/mag;
			Pos[i].z *= radius/mag;
			test = 1;
			for(int j = (NFe1 + NSi1 + NFe2); j < i; j++)
			{
				seperation = mag = sqrt((Pos[i].x-Pos[j].x)*(Pos[i].x-Pos[j].x) + (Pos[i].y-Pos[j].y)*(Pos[i].y-Pos[j].y) + (Pos[i].z-Pos[j].z)*(Pos[i].z-Pos[j].z));
				if(seperation < Diameter)
				{
					test = 0;
					repeatCount++;
					break;
				}
			}
		}
		Pos[i].w = 3.0;
		
		Vel[i].x = 0.0;
		Vel[i].y = 0.0;
		Vel[i].z = 0.0;
		Vel[i].w = MassSi;
	}
	printf("\nrepeat count body two = %d", repeatCount);
	
	//Setting the body one's center of mass location
	for(int i = (NFe1 + NSi1); i < N; i++)
	{
		Pos[i].x += InitialPosition2.x;
		Pos[i].y += InitialPosition2.y;
		Pos[i].z += InitialPosition2.z;
	}
	printf("\n************************************************** Initial bodies have been formed\n");
}

__global__ void getForcesSeperate(float4 *pos, float4 *vel, float4 *force, forceSeperateKernalConstantsStruct constant)
{
	int id, ids;
	int i,j;
	int inout;
    	float4 forceSum;
    	float4 posMe;
    	float4 velMe;
    	int test;
    	int materialSwitch;
    	float force_mag;
    	float4 dp;
    	float4 dv;
    	float r2;
    	float r;
    	float invr;
    
    	__shared__ float4 shPos[BLOCKSIZE];
    	__shared__ float4 shVel[BLOCKSIZE];
    
    	id = threadIdx.x + blockDim.x*blockIdx.x;
		
	forceSum.x = 0.0f;
	forceSum.y = 0.0f;
	forceSum.z = 0.0f;
		
	posMe.x = pos[id].x;
	posMe.y = pos[id].y;
	posMe.z = pos[id].z;
	
	velMe.x = vel[id].x;
	velMe.y = vel[id].y;
	velMe.z = vel[id].z;
		
	for(j = 0; j < gridDim.x; j++)
	{
		shPos[threadIdx.x] = pos[threadIdx.x + blockDim.x*j];
		shVel[threadIdx.x] = vel[threadIdx.x + blockDim.x*j];
		__syncthreads();
	   
		for(i = 0; i < blockDim.x; i++)	
		{
			ids = i + blockDim.x*j;
    			if((id < constant.boarder2 && ids < constant.boarder2) || (constant.boarder2 <= id && constant.boarder2 <= ids))
    			{
	    			if((id < constant.boarder2) && (ids < constant.boarder2)) materialSwitch = constant.boarder1;
	    			if((constant.boarder2 <= id) && (constant.boarder2 <= ids)) materialSwitch = constant.boarder3;
	    			
				dp.x = shPos[i].x - posMe.x;
				dp.y = shPos[i].y - posMe.y;
				dp.z = shPos[i].z - posMe.z;
				r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
				r = sqrt(r2);
				if(id == ids) invr = 0;
				else invr = 1.0f/r;

				test = 0;
				if(id < materialSwitch) test = 1;
				if(ids < materialSwitch) test++;
		
				if(test == 0) //silicate silicate force
				{
					if(1.0 <= r)
					{
						force_mag = 1.0/r2;  // G = 1 and mass of silicate elemnet =1
					}
					else if(constant.ShellBreakSi <= r)
					{
						force_mag = 1.0 - constant.KSiSi*(1.0 - r2);
					}
					else
					{
						dv.x = shVel[i].x - velMe.x;
						dv.y = shVel[i].y - velMe.y;
	 					dv.z = shVel[i].z - velMe.z;
						inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
						if(inout <= 0) 	force_mag  = 1.0 - constant.KSiSi*(1.0 - r2);
						else 		force_mag  = 1.0 - constant.KRSiSi*(1.0 - r2);
					}
				}
   	 			else if(test == 1) //Silicate iron force
				{
					if(1.0 <= r)
					{
						force_mag  = constant.GMassFeSi/r2;
					}
					else if(constant.ShellBreakFeSi1 <= r)
					{
						force_mag  = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
					}
					else if(constant.ShellBreakFeSi2 <= r)
					{
						dv.x = shVel[i].x - velMe.x;
						dv.y = shVel[i].y - velMe.y;
						dv.z = shVel[i].z - velMe.z;
						inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
						if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
		 				else 		force_mag = constant.GMassFeSi - constant.KRMix*(1.0 - r2);
					}
					else
					{
						dv.x = shVel[i].x - velMe.x;
						dv.y = shVel[i].y - velMe.y;
						dv.z = shVel[i].z - velMe.z;
						inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
						if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
						else 		force_mag = constant.GMassFeSi - constant.KRFeSi*(1.0 - r2);
		 			}
				}
				else //Iron iron force
				{
					if(1.0 <= r)
					{
						force_mag = constant.GMassFeFe/r2;
					}
					else if(constant.ShellBreakFe <= r)
					{
		    				force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
					}
					else
					{
						dv.x = shVel[i].x - velMe.x;
						dv.y = shVel[i].y - velMe.y;
						dv.z = shVel[i].z - velMe.z;
						inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
		   				if(inout <= 0) 	force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
		  				else 		force_mag = constant.GMassFeFe - constant.KRFeFe*(1.0 - r2);
					}
				}

				forceSum.x += force_mag*dp.x*invr;
				forceSum.y += force_mag*dp.y*invr;
				forceSum.z += force_mag*dp.z*invr;
			}
		}
		force[id].x = forceSum.x;
		force[id].y = forceSum.y;
		force[id].z = forceSum.z;
		__syncthreads();
	}
}

__global__ void moveBodiesSeperate(float4 *pos, float4 *vel, float4 * force, moveSeperateKernalConstantsStruct constant)
{
	float temp;
	int id;
	
    	id = threadIdx.x + blockDim.x*blockIdx.x;
    
	if(constant.boarder3 <= id) temp = constant.DtOverMassSi;
	else if(constant.boarder2 <= id) temp = constant.DtOverMassFe;
	else if(constant.boarder1 <= id) temp = constant.DtOverMassSi;
	else temp = constant.DtOverMassFe;

	vel[id].x += (force[id].x)*temp;
	vel[id].y += (force[id].y)*temp;
	vel[id].z += (force[id].z)*temp;

	pos[id].x += vel[id].x*constant.Dt;
	pos[id].y += vel[id].y*constant.Dt;
	pos[id].z += vel[id].z*constant.Dt;
}

__global__ void moveBodiesDampedSeperate(float4 *pos, float4 *vel, float4 * force, moveSeperateKernalConstantsStruct constant, float DampRateBody1, float DampRateBody2)
{
	float temp;
	float damp;
	int id;
	
    	id = threadIdx.x + blockDim.x*blockIdx.x;
 
	if(constant.boarder3 <= id) 
	{
		temp = constant.DtOverMassSi;
		damp = DampRateBody2;
	}
	else if(constant.boarder2 <= id) 
	{
		temp = constant.DtOverMassFe;
		damp = DampRateBody2;
	}
	else if(constant.boarder1 <= id) 
	{
		temp = constant.DtOverMassSi;
		damp = DampRateBody1;
	}
	else 
	{
		temp = constant.DtOverMassFe;
		damp = DampRateBody1;
	}
	
	vel[id].x += (force[id].x-damp*vel[id].x)*temp;
	vel[id].y += (force[id].y-damp*vel[id].y)*temp;
	vel[id].z += (force[id].z-damp*vel[id].z)*temp;

	pos[id].x += vel[id].x*constant.Dt;
	pos[id].y += vel[id].y*constant.Dt;
	pos[id].z += vel[id].z*constant.Dt;
}

__global__ void getForcesCollisionSingleGPU(float4 *pos, float4 *vel, float4 *force, forceCollisionKernalConstantsStruct constant)
{
	int id, ids;
	int inout;
	float4 forceSum;
	float4 posMe;
	float4 velMe;
	int test;
	float force_mag;
	float4 dp;
	float4 dv;
	float r2;
	float r;
	float invr;
	
	__shared__ float4 shPos[BLOCKSIZE];
	__shared__ float4 shVel[BLOCKSIZE];
	    
	id = threadIdx.x + blockDim.x*blockIdx.x;
	    
	forceSum.x = 0.0f;
	forceSum.y = 0.0f;
	forceSum.z = 0.0f;
		
	posMe.x = pos[id].x;
	posMe.y = pos[id].y;
	posMe.z = pos[id].z;
	
	velMe.x = vel[id].x;
	velMe.y = vel[id].y;
	velMe.z = vel[id].z;
		    
	for(int j=0; j < gridDim.x; j++)
	{
    		shPos[threadIdx.x] = pos[threadIdx.x + blockDim.x*j];
    		shVel[threadIdx.x] = vel[threadIdx.x + blockDim.x*j];
    		__syncthreads();
   
		for(int i=0; i < blockDim.x; i++)	
		{
			ids = i + blockDim.x*j;
		    	dp.x = shPos[i].x - posMe.x;
			dp.y = shPos[i].y - posMe.y;
			dp.z = shPos[i].z - posMe.z;
			r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
			r = sqrt(r2);
			if(id == ids) invr = 0;
			else invr = 1.0f/r;

		    	test = 0;
		    	if(id < constant.NFe) test = 1;
		    	if(ids < constant.NFe) test++;
	    
			if(test == 0) //Silicate silicate force
			{
				if(1.0 <= r)
				{
	    				force_mag = 1.0/r2; // G = 1 and mass of silicate elemnet =1
				}
				else if(constant.ShellBreakSi <= r)
				{
					force_mag = 1.0 - constant.KSiSi*(1.0 - r2); // because D = 1 G = 1 and mass of silicate = 1
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag  = 1.0 - constant.KSiSi*(1.0 - r2);
					else 		force_mag  = 1.0 - constant.KRSiSi*(1.0 - r2);
				}
	    		}
			else if(test == 1) //Silicate iron force
			{
				if(1.0 <= r)
				{
					force_mag  = constant.GMassFeSi/r2;
				}
				else if(constant.ShellBreakFeSi1 <= r)
				{
					force_mag  = constant.GMassFeSi -constant.KFeSi*(1.0 - r2);
				}
				else if(constant.ShellBreakFeSi2 <= r)
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
	 				else 		force_mag = constant.GMassFeSi - constant.KRMix*(1.0 - r2);
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
					else 		force_mag = constant.GMassFeSi - constant.KRFeSi*(1.0 - r2);
	 			}
			}
			else //Iron iron force
			{
				if(1.0 <= r)
				{
					force_mag = constant.GMassFeFe/r2;
				}
				else if(constant.ShellBreakFe <= r)
				{
	    			force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
	   				if(inout <= 0) 	force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
	  				else 		force_mag = constant.GMassFeFe - constant.KRFeFe*(1.0 - r2);
				}
			}

			forceSum.x += force_mag*dp.x*invr;
			forceSum.y += force_mag*dp.y*invr;
			forceSum.z += force_mag*dp.z*invr;
		}
		__syncthreads();
	}
	force[id].x = forceSum.x;
	force[id].y = forceSum.y;
	force[id].z = forceSum.z;
}

__global__ void moveBodiesCollisionSingleGPU(float4 *pos, float4 *vel, float4 * force, moveCollisionKernalConstantsStruct MoveCollisionConstant)
{
	float temp;
	int id;
    	id = threadIdx.x + blockDim.x*blockIdx.x;
    	if(id < MoveCollisionConstant.NFe) temp = MoveCollisionConstant.DtOverMassFe;
    	else temp = MoveCollisionConstant.DtOverMassSi;
	
	vel[id].x += (force[id].x)*temp;
	vel[id].y += (force[id].y)*temp;
	vel[id].z += (force[id].z)*temp;
	
	pos[id].x += vel[id].x*MoveCollisionConstant.Dt;
	pos[id].y += vel[id].y*MoveCollisionConstant.Dt;
	pos[id].z += vel[id].z*MoveCollisionConstant.Dt;
}

__global__ void getForcesCollisionDoubleGPU0(float4 *posFstHalf, float4 *posSndHalf, float4 *velFstHalf, float4 *velSndHalf,  float4 *forceFstHalf,  int N, forceCollisionKernalConstantsStruct constant)
{
	int id, ids;
	int i,j;
	int inout;
	float4 forceSum;
	float4 posMe;
	float4 velMe;
	int test;
	float force_mag;
	float4 dp;
	float4 dv;
	float r2;
	float r;
	float invr;

	__shared__ float4 shPos[BLOCKSIZE];
	__shared__ float4 shVel[BLOCKSIZE];

	id = threadIdx.x + blockDim.x*blockIdx.x;
	
	forceSum.x = 0.0f;
	forceSum.y = 0.0f;
	forceSum.z = 0.0f;

	posMe.x = posFstHalf[id].x;
	posMe.y = posFstHalf[id].y;
	posMe.z = posFstHalf[id].z;

	velMe.x = velFstHalf[id].x;
	velMe.y = velFstHalf[id].y;
	velMe.z = velFstHalf[id].z;
	    
	for(j=0;  j < gridDim.x;  j++)
	{
		shPos[threadIdx.x] = posFstHalf[threadIdx.x + blockDim.x*j];
		shVel[threadIdx.x]  = velFstHalf[threadIdx.x + blockDim.x*j];
		__syncthreads();

		for(i=0; i < blockDim.x; i++)	
		{
			ids = i + blockDim.x*j;
		    	dp.x = shPos[i].x - posMe.x;
			dp.y = shPos[i].y - posMe.y;
			dp.z = shPos[i].z - posMe.z;
			r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
			r = sqrt(r2);
			if(id == ids) invr = 0;
			else invr = 1.0f/r;

		    	test = 0;
		    	if(id < constant.NFe) test = 1;
		    	if(ids < constant.NFe) test++;   	
			    	
			if(test == 0) //Silicate silicate force
			{
				if(1.0 <= r)
				{
	    				force_mag = 1.0/r2; // G = 1 and mass of silicate elemnet =1
				}
				else if(constant.ShellBreakSi <= r)
				{
					force_mag = 1.0 - constant.KSiSi*(1.0 - r2); // because D = 1 G = 1 and mass of silicate = 1
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag  = 1.0 - constant.KSiSi*(1.0 - r2);
					else 		force_mag  = 1.0 - constant.KRSiSi*(1.0 - r2);
				}
	    		}
			else if(test == 1) //Silicate iron force
			{
				if(1.0 <= r)
				{
					force_mag  = constant.GMassFeSi/r2;
				}
				else if(constant.ShellBreakFeSi1 <= r)
				{
					force_mag  = constant.GMassFeSi -constant.KFeSi*(1.0 - r2);
				}
				else if(constant.ShellBreakFeSi2 <= r)
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
	 				else 		force_mag = constant.GMassFeSi - constant.KRMix*(1.0 - r2);
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
					else 		force_mag = constant.GMassFeSi - constant.KRFeSi*(1.0 - r2);
	 			}
			}
			else //Iron iron force
			{
				if(1.0 <= r)
				{
					force_mag = constant.GMassFeFe/r2;
				}
				else if(constant.ShellBreakFe <= r)
				{
	    			force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
	   				if(inout <= 0) 	force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
	  				else 		force_mag = constant.GMassFeFe - constant.KRFeFe*(1.0 - r2);
				}
			}

			forceSum.x += force_mag*dp.x*invr;
			forceSum.y += force_mag*dp.y*invr;
			forceSum.z += force_mag*dp.z*invr;
		}
		__syncthreads();
	}
	
	for(j=0; j < gridDim.x; j++)
	{
		shPos[threadIdx.x] = posSndHalf[threadIdx.x + blockDim.x*j];
		shVel[threadIdx.x] = velSndHalf[threadIdx.x + blockDim.x*j];
		__syncthreads();

		for(i=0; i < blockDim.x; i++)	
		{
			ids = i + blockDim.x*j;
		    	dp.x = shPos[i].x - posMe.x;
			dp.y = shPos[i].y - posMe.y;
			dp.z = shPos[i].z - posMe.z;
			r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
			r = sqrt(r2);
		 	invr = 1.0f/r;

		    	test = 0;
		    	if(id  < constant.NFe) test = 1;
		    	if(ids+(N/2) < constant.NFe) test++;   	
			    	
			if(test == 0) //Silicate silicate force
			{
				if(1.0 <= r)
				{
	    				force_mag = 1.0/r2; // G = 1 and mass of silicate elemnet =1
				}
				else if(constant.ShellBreakSi <= r)
				{
					force_mag = 1.0 - constant.KSiSi*(1.0 - r2); // because D = 1 G = 1 and mass of silicate = 1
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag  = 1.0 - constant.KSiSi*(1.0 - r2);
					else 		force_mag  = 1.0 - constant.KRSiSi*(1.0 - r2);
				}
	    		}
			else if(test == 1) //Silicate iron force
			{
				if(1.0 <= r)
				{
					force_mag  = constant.GMassFeSi/r2;
				}
				else if(constant.ShellBreakFeSi1 <= r)
				{
					force_mag  = constant.GMassFeSi -constant.KFeSi*(1.0 - r2);
				}
				else if(constant.ShellBreakFeSi2 <= r)
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
	 				else 		force_mag = constant.GMassFeSi - constant.KRMix*(1.0 - r2);
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
					else 		force_mag = constant.GMassFeSi - constant.KRFeSi*(1.0 - r2);
	 			}
			}
			else //Iron iron force
			{
				if(1.0 <= r)
				{
					force_mag = constant.GMassFeFe/r2;
				}
				else if(constant.ShellBreakFe <= r)
				{
	    			force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
	   				if(inout <= 0) 	force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
	  				else 		force_mag = constant.GMassFeFe - constant.KRFeFe*(1.0 - r2);
				}
			}

			forceSum.x += force_mag*dp.x*invr;
			forceSum.y += force_mag*dp.y*invr;
			forceSum.z += force_mag*dp.z*invr;
		}
		__syncthreads();
	}

	forceFstHalf[id].x = forceSum.x;
	forceFstHalf[id].y = forceSum.y;
	forceFstHalf[id].z = forceSum.z;
}

__global__ void getForcesCollisionDoubleGPU1(float4 *posFstHalf, float4 *posSndHalf, float4 *velFstHalf, float4 *velSndHalf,  float4 *forceSndHalf,  int N, forceCollisionKernalConstantsStruct constant)
{
	int id, ids;
	int i,j;
	int inout;
	float4 forceSum;
	float4 posMe;
	float4 velMe;
	int test;
	float force_mag;
	float4 dp;
	float4 dv;
	float r2;
	float r;
	float invr;

	__shared__ float4 shPos[BLOCKSIZE];
	__shared__ float4 shVel[BLOCKSIZE];

	id = threadIdx.x + blockDim.x*blockIdx.x;
	
	forceSum.x = 0.0f;
	forceSum.y = 0.0f;
	forceSum.z = 0.0f;

	posMe.x = posSndHalf[id].x;
	posMe.y = posSndHalf[id].y;
	posMe.z = posSndHalf[id].z;

	velMe.x = velSndHalf[id].x;
	velMe.y = velSndHalf[id].y;
	velMe.z = velSndHalf[id].z;
	    
	for(j=0;  j < gridDim.x;  j++)
	{
		shPos[threadIdx.x] = posFstHalf[threadIdx.x + blockDim.x*j];
		shVel[threadIdx.x]  = velFstHalf[threadIdx.x + blockDim.x*j];
		__syncthreads();

		for(i=0; i < blockDim.x; i++)	
		{
			ids = i + blockDim.x*j;
		    	dp.x = shPos[i].x - posMe.x;
			dp.y = shPos[i].y - posMe.y;
			dp.z = shPos[i].z - posMe.z;
			r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
			r = sqrt(r2);
			invr = 1.0f/r;

		    	test = 0;
		    	if(id + (N/2) < constant.NFe) test = 1;
		    	if(ids < constant.NFe) test++;   	
			    	
			if(test == 0) //Silicate silicate force
			{
				if(1.0 <= r)
				{
	    				force_mag = 1.0/r2; // G = 1 and mass of silicate elemnet =1
				}
				else if(constant.ShellBreakSi <= r)
				{
					force_mag = 1.0 - constant.KSiSi*(1.0 - r2); // because D = 1 G = 1 and mass of silicate = 1
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag  = 1.0 - constant.KSiSi*(1.0 - r2);
					else 		force_mag  = 1.0 - constant.KRSiSi*(1.0 - r2);
				}
	    		}
			else if(test == 1) //Silicate iron force
			{
				if(1.0 <= r)
				{
					force_mag  = constant.GMassFeSi/r2;
				}
				else if(constant.ShellBreakFeSi1 <= r)
				{
					force_mag  = constant.GMassFeSi -constant.KFeSi*(1.0 - r2);
				}
				else if(constant.ShellBreakFeSi2 <= r)
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
	 				else 		force_mag = constant.GMassFeSi - constant.KRMix*(1.0 - r2);
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
					else 		force_mag = constant.GMassFeSi - constant.KRFeSi*(1.0 - r2);
	 			}
			}
			else //Iron iron force
			{
				if(1.0 <= r)
				{
					force_mag = constant.GMassFeFe/r2;
				}
				else if(constant.ShellBreakFe <= r)
				{
	    			force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
	   				if(inout <= 0) 	force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
	  				else 		force_mag = constant.GMassFeFe - constant.KRFeFe*(1.0 - r2);
				}
			}

			forceSum.x += force_mag*dp.x*invr;
			forceSum.y += force_mag*dp.y*invr;
			forceSum.z += force_mag*dp.z*invr;
		}
		__syncthreads();
	}
	
	for(j=0; j < gridDim.x; j++)
	{
		shPos[threadIdx.x] = posSndHalf[threadIdx.x + blockDim.x*j];
		shVel[threadIdx.x] = velSndHalf[threadIdx.x + blockDim.x*j];
		__syncthreads();

		for(i=0; i < blockDim.x; i++)	
		{
			ids = i + blockDim.x*j ;
		    	dp.x = shPos[i].x - posMe.x;
			dp.y = shPos[i].y - posMe.y;
			dp.z = shPos[i].z - posMe.z;
			r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
			r = sqrt(r2);
			if(id == ids) invr = 0;
			else invr = 1.0f/r;

		    	test = 0;
		    	if(id + (N/2) < constant.NFe) test = 1;
		    	if(ids+(N/2) < constant.NFe) test++;   	
			    	
			if(test == 0) //Silicate silicate force
			{
				if(1.0 <= r)
				{
	    				force_mag = 1.0/r2; // G = 1 and mass of silicate elemnet =1
				}
				else if(constant.ShellBreakSi <= r)
				{
					force_mag = 1.0 - constant.KSiSi*(1.0 - r2); // because D = 1 G = 1 and mass of silicate = 1
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag  = 1.0 - constant.KSiSi*(1.0 - r2);
					else 		force_mag  = 1.0 - constant.KRSiSi*(1.0 - r2);
				}
	    		}
			else if(test == 1) //Silicate iron force
			{
				if(1.0 <= r)
				{
					force_mag  = constant.GMassFeSi/r2;
				}
				else if(constant.ShellBreakFeSi1 <= r)
				{
					force_mag  = constant.GMassFeSi -constant.KFeSi*(1.0 - r2);
				}
				else if(constant.ShellBreakFeSi2 <= r)
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
	 				else 		force_mag = constant.GMassFeSi - constant.KRMix*(1.0 - r2);
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
					if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
					else 		force_mag = constant.GMassFeSi - constant.KRFeSi*(1.0 - r2);
	 			}
			}
			else //Iron iron force
			{
				if(1.0 <= r)
				{
					force_mag = constant.GMassFeFe/r2;
				}
				else if(constant.ShellBreakFe <= r)
				{
	    			force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
				}
				else
				{
					dv.x = shVel[i].x - velMe.x;
					dv.y = shVel[i].y - velMe.y;
					dv.z = shVel[i].z - velMe.z;
					inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
	   				if(inout <= 0) 	force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
	  				else 		force_mag = constant.GMassFeFe - constant.KRFeFe*(1.0 - r2);
				}
			}

			forceSum.x += force_mag*dp.x*invr;
			forceSum.y += force_mag*dp.y*invr;
			forceSum.z += force_mag*dp.z*invr;
		}
		__syncthreads();
	}

	forceSndHalf[id].x = forceSum.x;
	forceSndHalf[id].y = forceSum.y;
	forceSndHalf[id].z = forceSum.z;
}
		
__global__ void moveBodiesCollisionDoubleGPU0(float4 *posFstHalf,   float4 *velFstHalf,  float4 * forceFstHalf,  int N, moveCollisionKernalConstantsStruct constant)
{
	float temp;
	int id;
	id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id < constant.NFe) temp = constant.DtOverMassFe;
    	else temp = constant.DtOverMassSi;
	
	velFstHalf[id].x += (forceFstHalf[id].x)*temp;
	velFstHalf[id].y += (forceFstHalf[id].y)*temp;
	velFstHalf[id].z += (forceFstHalf[id].z)*temp;

	posFstHalf[id].x += velFstHalf[id].x*constant.Dt;
	posFstHalf[id].y += velFstHalf[id].y*constant.Dt;
	posFstHalf[id].z += velFstHalf[id].z*constant.Dt;
}

__global__ void moveBodiesCollisionDoubleGPU1(float4 *posSndHalf,  float4 *velSndHalf,  float4 * forceSndHalf,  int N, moveCollisionKernalConstantsStruct constant)
{
	float temp;
	int id;
	id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id + (N/2) < constant.NFe) temp = constant.DtOverMassFe;
    	else temp = constant.DtOverMassSi;
	
	velSndHalf[id].x += (forceSndHalf[id].x)*temp;
	velSndHalf[id].y += (forceSndHalf[id].y)*temp;
	velSndHalf[id].z += (forceSndHalf[id].z)*temp;

	posSndHalf[id].x += velSndHalf[id].x*constant.Dt;
	posSndHalf[id].y += velSndHalf[id].y*constant.Dt;
	posSndHalf[id].z += velSndHalf[id].z*constant.Dt;
}

float3 getCenterOfMassSeperate(int scope)
{
	float totalMass;
	float assumeZero = 0.0000001;
	float3 centerOfMass;
	
	centerOfMass.x = 0.0f;
	centerOfMass.y = 0.0f;
	centerOfMass.z = 0.0f;
	
	if(scope == 0) //entire system
	{
		totalMass = MassOfBody1 + MassOfBody2;
		if(totalMass < assumeZero) return(centerOfMass);
		
		for(int i = 0; i < NFe1; i++)
		{
	    		centerOfMass.x += Pos[i].x*MassFe;
			centerOfMass.y += Pos[i].y*MassFe;
			centerOfMass.z += Pos[i].z*MassFe;
		}
		for(int i = NFe1; i < NFe1 + NSi1; i++)
		{
	    		centerOfMass.x += Pos[i].x*MassSi;
			centerOfMass.y += Pos[i].y*MassSi;
			centerOfMass.z += Pos[i].z*MassSi;
		}
		for(int i = NFe1 + NSi1; i < NFe1 + NSi1 + NFe2; i++)
		{
	    		centerOfMass.x += Pos[i].x*MassFe;
			centerOfMass.y += Pos[i].y*MassFe;
			centerOfMass.z += Pos[i].z*MassFe;
		}
		for(int i = NFe1 + NSi1 + NFe2; i < N; i++)
		{
	    		centerOfMass.x += Pos[i].x*MassSi;
			centerOfMass.y += Pos[i].y*MassSi;
			centerOfMass.z += Pos[i].z*MassSi;
		}
	}
	else if(scope == 1) //body1
	{
		totalMass = MassOfBody1;
		if(totalMass < assumeZero) return(centerOfMass);
		
		for(int i = 0; i < NFe1; i++)
		{
	    		centerOfMass.x += Pos[i].x*MassFe;
			centerOfMass.y += Pos[i].y*MassFe;
			centerOfMass.z += Pos[i].z*MassFe;
		}
		for(int i = NFe1; i < NFe1 + NSi1; i++)
		{
	    		centerOfMass.x += Pos[i].x*MassSi;
			centerOfMass.y += Pos[i].y*MassSi;
			centerOfMass.z += Pos[i].z*MassSi;
		}
	}
	else if(scope == 2) //body2
	{
		totalMass = MassOfBody2;
		if(totalMass < assumeZero) return(centerOfMass);
		
		for(int i = NFe1 + NSi1; i < NFe1 + NSi1 + NFe2; i++)
		{
	    		centerOfMass.x += Pos[i].x*MassFe;
			centerOfMass.y += Pos[i].y*MassFe;
			centerOfMass.z += Pos[i].z*MassFe;
		}
		for(int i = NFe1 + NSi1 + NFe2; i < N; i++)
		{
	    		centerOfMass.x += Pos[i].x*MassSi;
			centerOfMass.y += Pos[i].y*MassSi;
			centerOfMass.z += Pos[i].z*MassSi;
		}
	}
	else
	{
		printf("\nTSU Error: In getCenterOfMassSeperate function scope invalid\n");
		exit(0);
	}
	
	centerOfMass.x /= totalMass;
	centerOfMass.y /= totalMass;
	centerOfMass.z /= totalMass;
	return(centerOfMass);
}

float3 getLinearVelocitySeperate(int scope)
{
	double totalMass;
	float assumeZero = 0.0000001;
	float3 linearVelocity;
	
	linearVelocity.x = 0.0f;
	linearVelocity.y = 0.0f;
	linearVelocity.z = 0.0f;
	
	if(scope == 0) //Entire system
	{
		totalMass = MassOfBody1 + MassOfBody2;
		if(totalMass < assumeZero) return(linearVelocity);
		
		for(int i = 0; i < NFe1; i++)
		{
	    		linearVelocity.x += Vel[i].x*MassFe;
			linearVelocity.y += Vel[i].y*MassFe;
			linearVelocity.z += Vel[i].z*MassFe;
		}
		for(int i = NFe1; i < NFe1 + NSi1; i++)
		{
	    		linearVelocity.x += Vel[i].x*MassSi;
			linearVelocity.y += Vel[i].y*MassSi;
			linearVelocity.z += Vel[i].z*MassSi;
		}

		for(int i = NFe1 + NSi1; i < NFe1 + NSi1 + NFe2; i++)
		{
	    		linearVelocity.x += Vel[i].x*MassFe;
			linearVelocity.y += Vel[i].y*MassFe;
			linearVelocity.z += Vel[i].z*MassFe;
		}
		for(int i = NFe1 + NSi1 + NFe2; i < N; i++)
		{
	    		linearVelocity.x += Vel[i].x*MassSi;
			linearVelocity.y += Vel[i].y*MassSi;
			linearVelocity.z += Vel[i].z*MassSi;
		}
	}
	else if(scope == 1) //body1
	{
		totalMass = MassOfBody1;
		if(totalMass < assumeZero) return(linearVelocity);
		
		for(int i = 0; i < NFe1; i++)
		{
	    		linearVelocity.x += Vel[i].x*MassFe;
			linearVelocity.y += Vel[i].y*MassFe;
			linearVelocity.z += Vel[i].z*MassFe;
		}
		for(int i = NFe1; i < NFe1 + NSi1; i++)
		{
	    		linearVelocity.x += Vel[i].x*MassSi;
			linearVelocity.y += Vel[i].y*MassSi;
			linearVelocity.z += Vel[i].z*MassSi;
		}
	}
	else if (scope == 2) //body2
	{
		totalMass = MassOfBody2;
		if(totalMass < assumeZero) return(linearVelocity);
		
		for(int i = NFe1 + NSi1; i < NFe1 + NSi1 + NFe2; i++)
		{
	    		linearVelocity.x += Vel[i].x*MassFe;
			linearVelocity.y += Vel[i].y*MassFe;
			linearVelocity.z += Vel[i].z*MassFe;
		}
		for(int i = NFe1 + NSi1 + NFe2; i < N; i++)
		{
	    		linearVelocity.x += Vel[i].x*MassSi;
			linearVelocity.y += Vel[i].y*MassSi;
			linearVelocity.z += Vel[i].z*MassSi;
		}
	}
	else
	{
		printf("\nTSU Error: In getLinearVelocitySeperate function scope invalid\n");
		exit(0);
	}
	
	linearVelocity.x /= totalMass;
	linearVelocity.y /= totalMass;
	linearVelocity.z /= totalMass;
	return(linearVelocity);
}

float3 getAngularMomentumSeperate(int scope, float3 center, float3 velocity)
{
	float3 angularMomentum;
	float3 r;
	float3 v;
	
	angularMomentum.x = 0.0f;
	angularMomentum.y = 0.0f;
	angularMomentum.z = 0.0f;
	
	if(scope == 0) //entire system
	{	
		for(int i = 0; i < NFe1; i++)
		{
			r.x = Pos[i].x - center.x;
			r.y = Pos[i].y - center.y;
			r.z = Pos[i].z - center.z;
		
			v.x = Vel[i].x - velocity.x;
			v.y = Vel[i].y - velocity.y;
			v.z = Vel[i].z - velocity.z;
		
			angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassFe;
			angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassFe;
			angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassFe;
		}
		for(int i = NFe1; i < NFe1 + NSi1; i++)
		{
			r.x = Pos[i].x - center.x;
			r.y = Pos[i].y - center.y;
			r.z = Pos[i].z - center.z;
		
			v.x = Vel[i].x - velocity.x;
			v.y = Vel[i].y - velocity.y;
			v.z = Vel[i].z - velocity.z;
		
			angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassSi;
			angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassSi;
			angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassSi;
		}
		for(int i = NFe1 + NSi1; i < NFe1 + NSi1 + NFe2; i++)
		{
			r.x = Pos[i].x - center.x;
			r.y = Pos[i].y - center.y;
			r.z = Pos[i].z - center.z;
		
			v.x = Vel[i].x - velocity.x;
			v.y = Vel[i].y - velocity.y;
			v.z = Vel[i].z - velocity.z;
		
			angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassFe;
			angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassFe;
			angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassFe;
		}
		for(int i = NFe1 + NSi1 + NFe2; i < N; i++)
		{
			r.x = Pos[i].x - center.x;
			r.y = Pos[i].y - center.y;
			r.z = Pos[i].z - center.z;
		
			v.x = Vel[i].x - velocity.x;
			v.y = Vel[i].y - velocity.y;
			v.z = Vel[i].z - velocity.z;
		
			angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassSi;
			angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassSi;
			angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassSi;
		}
	}
	else if(scope == 1) //body1
	{	
		for(int i = 0; i < NFe1; i++)
		{
			r.x = Pos[i].x - center.x;
			r.y = Pos[i].y - center.y;
			r.z = Pos[i].z - center.z;
		
			v.x = Vel[i].x - velocity.x;
			v.y = Vel[i].y - velocity.y;
			v.z = Vel[i].z - velocity.z;
		
			angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassFe;
			angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassFe;
			angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassFe;
		}
		for(int i = NFe1; i < NFe1 + NSi1; i++)
		{
			r.x = Pos[i].x - center.x;
			r.y = Pos[i].y - center.y;
			r.z = Pos[i].z - center.z;
		
			v.x = Vel[i].x - velocity.x;
			v.y = Vel[i].y - velocity.y;
			v.z = Vel[i].z - velocity.z;
		
			angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassSi;
			angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassSi;
			angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassSi;
		}
	}
	else if(scope == 2) //body2
	{
		for(int i = NFe1 + NSi1; i < NFe1 + NSi1 + NFe2; i++)
		{
			r.x = Pos[i].x - center.x;
			r.y = Pos[i].y - center.y;
			r.z = Pos[i].z - center.z;
		
			v.x = Vel[i].x - velocity.x;
			v.y = Vel[i].y - velocity.y;
			v.z = Vel[i].z - velocity.z;
		
			angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassFe;
			angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassFe;
			angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassFe;
		}
		for(int i = NFe1 + NSi1 + NFe2; i < N; i++)
		{
			r.x = Pos[i].x - center.x;
			r.y = Pos[i].y - center.y;
			r.z = Pos[i].z - center.z;
		
			v.x = Vel[i].x - velocity.x;
			v.y = Vel[i].y - velocity.y;
			v.z = Vel[i].z - velocity.z;
		
			angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassSi;
			angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassSi;
			angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassSi;
		}
	}
	else
	{
		printf("\nTSU Error: In getAngularMomentumSeperate function scope invalid\n");
		exit(0);
	}

	return(angularMomentum);
}

void setBodyPositionSeperate(int bodyId, float x, float y, float z)
{
	int	start, stop;
	
	if(bodyId == 1)
	{
		start = 0;
		stop = NFe1 + NSi1;
	}
	else if(bodyId == 2)
	{
		start = NFe1 + NSi1;
		stop = N;
	}
	else 
	{
		printf("\nTSU Error: in setBodyPositionSeperate function bodyId invalid\n");
		exit(0);
	}
	
	float3 centerOfMass = getCenterOfMassSeperate(bodyId); 
	
	for(int i = start; i < stop; i++)
	{
		Pos[i].x += x - centerOfMass.x;
		Pos[i].y += y - centerOfMass.y;
		Pos[i].z += z - centerOfMass.z;
	}	
}

void setBodyVelocitySeperate(int bodyId, float vx, float vy, float vz)
{
	int	start, stop;
	
	if(bodyId == 1)
	{
		start = 0;
		stop = NFe1 + NSi1;
	}
	else if(bodyId == 2)
	{
		start = NFe1 + NSi1;
		stop = N;
	}
	else 
	{
		printf("\nTSU Error: in setBodyVelocitySeperate invalid bodyId\n");
		exit(0);
	}
	
	float3 RandomlinearVelocity = getLinearVelocitySeperate(bodyId); 
	
	for(int i = start; i < stop; i++)
	{
		Vel[i].x += vx - RandomlinearVelocity.x;
		Vel[i].y += vy - RandomlinearVelocity.y;
		Vel[i].z += vz - RandomlinearVelocity.z;
	}	
}

void spinBodySeperate(int bodyId, float4 spinVector)
{
	float3 	r;  			//vector from center of mass to the position vector
	float3 	centerOfMass;
	float3	n;			//Unit vector perpendicular to the plane of spin
	float 	mag;
	float 	assumeZero = 0.0000001;
	int	start, stop;
	
	if(bodyId == 1)
	{
		start = 0;
		stop = NFe1 + NSi1;
	}
	else
	{
		start = NFe1 + NSi1;
		stop = N;
	}
	
	//Making sure the spin vector is a unit vector
	mag = sqrt(spinVector.x*spinVector.x + spinVector.y*spinVector.y + spinVector.z*spinVector.z);
	if(assumeZero < mag)
	{
		spinVector.x /= mag;
		spinVector.y /= mag;
		spinVector.z /= mag;
	}
	else 
	{
		printf("\nTSU Error: In spinBodySeperate. The spin direction vector is zero.\n");
		exit(0);
	}
	
	centerOfMass = getCenterOfMassSeperate(bodyId);
	for(int i = start; i < stop; i++)
	{
		//Creating a vector from the center of mass to the point
		r.x = Pos[i].x - centerOfMass.x;
		r.y = Pos[i].y - centerOfMass.y;
		r.z = Pos[i].z - centerOfMass.z;
		float magsquared = r.x*r.x + r.y*r.y + r.z*r.z;
		float spinDota = spinVector.x*r.x + spinVector.y*r.y + spinVector.z*r.z;
		float perpendicularDistance = sqrt(magsquared - spinDota*spinDota);
		float perpendicularVelocity = spinVector.w*2.0*Pi*perpendicularDistance;
		
		//finding unit vector perpendicular to both the position vector and the spin vector
		n.x =  (spinVector.y*r.z - spinVector.z*r.y);
		n.y = -(spinVector.x*r.z - spinVector.z*r.x);
		n.z =  (spinVector.x*r.y - spinVector.y*r.x);
		mag = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
		if(mag != 0.0)
		{
			n.x /= mag;
			n.y /= mag;
			n.z /= mag;
				
			//Spining the element
			Vel[i].x += perpendicularVelocity*n.x;
			Vel[i].y += perpendicularVelocity*n.y;
			Vel[i].z += perpendicularVelocity*n.z;
		}
	}		
}

double vectorMagnitude(float3 v)
{
	return(sqrt(v.x*v.x + v.y*v.y + v.z*v.z));
}

void recordStatsOfCreatedBodies()
{
	float radiusOfBody;
	float massOfBody;
	float3 r;
	double mag, d;
	
	float3 centerOfMass;
	float3 linearVelocity;
	float3 angularMomentum;
	
	double lengthConvertion = UnitLength;
	double massConvertion = UnitMass;
	double velocityConvertion = UnitLength/UnitTime;
	double AngularMomentumConvertion = (UnitMass*UnitLength*UnitLength)/(UnitTime);
	
	fprintf(RunStatsFile, "\n\n\n*****************************************************************************************************\n");
	fprintf(RunStatsFile, "\nThe follow are the statistics of the system right before they are released to collide in real world units\n");

	fprintf(RunStatsFile, "\n\n***** Stats for the univeral system *****\n");
	centerOfMass = getCenterOfMassSeperate(0);
	fprintf(RunStatsFile, "\nThe center of mass = (%f, %f, %f) Kilometers from (0, 0, 0)\n", centerOfMass.x*lengthConvertion, centerOfMass.y*lengthConvertion, centerOfMass.z*lengthConvertion);
	
	linearVelocity = getLinearVelocitySeperate(0);
	fprintf(RunStatsFile, "\nThe average linear velocity = (%f, %f, %f)", linearVelocity.x*velocityConvertion, linearVelocity.y*velocityConvertion, linearVelocity.z*velocityConvertion);
	mag = vectorMagnitude(linearVelocity);
	fprintf(RunStatsFile, "\nThe magitude of the avergae linear velocity = %f Kilometers/second\n", mag*velocityConvertion);
	
	angularMomentum = getAngularMomentumSeperate(0, getCenterOfMassSeperate(0), getLinearVelocitySeperate(0));
	fprintf(RunStatsFile, "\nThe angular momentum = (%e, %e, %e)", angularMomentum.x*AngularMomentumConvertion, angularMomentum.y*AngularMomentumConvertion, angularMomentum.z*AngularMomentumConvertion);
	mag = vectorMagnitude(angularMomentum);
	fprintf(RunStatsFile, "\nThe magitude of the angular momentum = %e Kilograms*kilometers*kilometers/second\n", mag*AngularMomentumConvertion);
	
	fprintf(RunStatsFile, "\n\n***** Stats for Body1 *****\n");
	centerOfMass = getCenterOfMassSeperate(1);
	
	radiusOfBody = 0.0;
	massOfBody = 0.0;
	for(int i = 0; i < NFe1; i++)
	{
		r.x = Pos[i].x - centerOfMass.x;
		r.y = Pos[i].y - centerOfMass.y;
		r.z = Pos[i].z - centerOfMass.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(d > radiusOfBody) radiusOfBody = d;
		
		massOfBody += MassFe;
	}
	
	for(int i = NFe1; i < NSi1; i++)
	{
		r.x = Pos[i].x - centerOfMass.x;
		r.y = Pos[i].y - centerOfMass.y;
		r.z = Pos[i].z - centerOfMass.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(d > radiusOfBody) radiusOfBody = d;
		
		massOfBody += MassSi;
	}
	
	fprintf(RunStatsFile, "\nMass =  %e Kilograms\n", massOfBody*massConvertion);
	fprintf(RunStatsFile, "\nRadius =  %f Kilometers\n", radiusOfBody*lengthConvertion);
	
	fprintf(RunStatsFile, "\nThe center of mass = (%f, %f, %f) Kilometers from (0, 0, 0)\n", centerOfMass.x*lengthConvertion, centerOfMass.y*lengthConvertion, centerOfMass.z*lengthConvertion);
	
	linearVelocity = getLinearVelocitySeperate(1);
	fprintf(RunStatsFile, "\nThe average linear velocity = (%f, %f, %f)", linearVelocity.x*velocityConvertion, linearVelocity.y*velocityConvertion, linearVelocity.z*velocityConvertion);
	mag = vectorMagnitude(linearVelocity);
	fprintf(RunStatsFile, "\nThe magitude of the avergae linear velocity = %f Kilometers/second\n", mag*velocityConvertion);
	
	angularMomentum = getAngularMomentumSeperate(1, getCenterOfMassSeperate(1), getLinearVelocitySeperate(1));
	fprintf(RunStatsFile, "\nThe angular momentum = (%e, %e, %e)", angularMomentum.x*AngularMomentumConvertion, angularMomentum.y*AngularMomentumConvertion, angularMomentum.z*AngularMomentumConvertion);
	mag = vectorMagnitude(angularMomentum);
	fprintf(RunStatsFile, "\nThe magitude of the angular momentum = %e Kilograms*kilometers*kilometers/second\n", mag*AngularMomentumConvertion);
	
	fprintf(RunStatsFile, "\n\n***** Stats for Body2 *****\n");
	centerOfMass = getCenterOfMassSeperate(2);
	
	radiusOfBody = 0.0;
	massOfBody = 0.0;
	for(int i = NFe1 + NSi1; i < NFe1 + NSi1 + NFe2; i++)
	{
		r.x = Pos[i].x - centerOfMass.x;
		r.y = Pos[i].y - centerOfMass.y;
		r.z = Pos[i].z - centerOfMass.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(d > radiusOfBody) radiusOfBody = d;
		
		massOfBody += MassFe;
	}
	
	for(int i = NFe1 + NSi1 + NFe2; i < N; i++)
	{
		r.x = Pos[i].x - centerOfMass.x;
		r.y = Pos[i].y - centerOfMass.y;
		r.z = Pos[i].z - centerOfMass.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(d > radiusOfBody) radiusOfBody = d;
		
		massOfBody += MassSi;
	}
	
	fprintf(RunStatsFile, "\nMass =  %e Kilograms\n", massOfBody*massConvertion);
	fprintf(RunStatsFile, "\nRadius =  %f Kilometers\n", radiusOfBody*lengthConvertion);
	
	fprintf(RunStatsFile, "\nThe center of mass = (%f, %f, %f) Kilometers from (0, 0, 0)\n", centerOfMass.x*lengthConvertion, centerOfMass.y*lengthConvertion, centerOfMass.z*lengthConvertion);
	
	linearVelocity = getLinearVelocitySeperate(2);
	fprintf(RunStatsFile, "\nThe average linear velocity = (%f, %f, %f)", linearVelocity.x*velocityConvertion, linearVelocity.y*velocityConvertion, linearVelocity.z*velocityConvertion);
	mag = vectorMagnitude(linearVelocity);
	fprintf(RunStatsFile, "\nThe magitude of the avergae linear velocity = %f Kilometers/second\n", mag*velocityConvertion);
	
	angularMomentum = getAngularMomentumSeperate(2, getCenterOfMassSeperate(2), getLinearVelocitySeperate(2));
	fprintf(RunStatsFile, "\nThe angular momentum = (%e, %e, %e)", angularMomentum.x*AngularMomentumConvertion, angularMomentum.y*AngularMomentumConvertion, angularMomentum.z*AngularMomentumConvertion);
	mag = vectorMagnitude(angularMomentum);
	fprintf(RunStatsFile, "\nThe magitude of the angular momentum = %e Kilograms*kilometers*kilometers/second\n", mag*AngularMomentumConvertion);
}

void recordStartPosVelOfCreatedBodiesSeperate()
{
	cudaMemcpy( Pos, Pos_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Pos1");
	cudaMemcpy( Vel, Vel_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Vel");
	
	fwrite(Pos, sizeof(float4), N, StartPosAndVelFile);
	fwrite(Vel, sizeof(float4), N, StartPosAndVelFile);
}

int findEarthAndMoon()
{
	int groupId[N], used[N];
	float mag, dx, dy, dz;
	float touch = Diameter*1.5;
	int groupNumber, numberOfGroups;
	int k;
	
	for(int i = 0; i < N; i++)
	{
		groupId[i] = -1;
		used[i] = 0;
	}
	
	groupNumber = 0;
	for(int i = 0; i < N; i++)
	{
		if(groupId[i] == -1)
		{
			groupId[i] = groupNumber;
			//find all from this group
			k = i;
			while(k < N)
			{
				if(groupId[k] == groupNumber && used[k] == 0)
				{
					for(int j = i; j < N; j++)
					{
						dx = Pos[k].x - Pos[j].x;
						dy = Pos[k].y - Pos[j].y;
						dz = Pos[k].z - Pos[j].z;
						mag = sqrt(dx*dx + dy*dy + dz*dz);
						if(mag < touch)
						{
							groupId[j] = groupNumber;
						}
					}
					used[k] = 1;
					k = i;
				}
				else k++;	
			}
			
		}
		groupNumber++;
	}
	numberOfGroups = groupNumber;
	
	if(numberOfGroups == 1)
	{
		printf("\n No Moon found\n");
	}
	
	int count;
	int *groupSize = (int *)malloc(numberOfGroups*sizeof(int));
	for(int i = 0; i < numberOfGroups; i++)
	{
		count = 0;
		for(int j = 0; j < N; j++)
		{
			if(i == groupId[j]) count++;
		}
		groupSize[i] = count;
	}
	
	int earthGroupId = -1;
	NumberOfEarthElements = 0;
	for(int i = 0; i < numberOfGroups; i++)
	{
		if(groupSize[i] > NumberOfEarthElements)
		{
			NumberOfEarthElements = groupSize[i];
			earthGroupId = i;
		}
	}
	
	int moonGroupId = -1;
	NumberOfMoonElements = 0;
	for(int i = 0; i < numberOfGroups; i++)
	{
		if(groupSize[i] > NumberOfMoonElements && i != earthGroupId)
		{
			NumberOfMoonElements = groupSize[i];
			moonGroupId = i;
		}
	}
	
	free(groupSize);
	EarthIndex = (int *)malloc(NumberOfEarthElements*sizeof(int));
	MoonIndex = (int *)malloc(NumberOfMoonElements*sizeof(int));
	
	int earthCount = 0;
	int moonCount = 0;
	for(int j = 0; j < N; j++)
	{
		if(groupId[j] == earthGroupId) 
		{
			EarthIndex[earthCount] = j;
			earthCount++;
		}
		else if(groupId[j] == moonGroupId)  
		{
			MoonIndex[moonCount] = j;
			moonCount++;
		}
	}
	
	return(1);	
}

float getMassCollision(int scope)
{
	float mass = 0.0;
	
	if(scope == 0) // entire system
	{
		for(int i = 0; i < N; i++)
		{
			if(i < NFe) mass += MassFe;
			else mass += MassSi;
		}
	}
	else if(scope == 1) // earth-moon syatem
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe) mass += MassFe;
			else mass += MassSi;
		}
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe) mass += MassFe;
			else mass += MassSi;
		}
	}
	else if(scope == 2) // earth
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe) mass += MassFe;
			else mass += MassSi;
		}
	}
	else if(scope == 3) // moon
	{
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe) mass += MassFe;
			else mass += MassSi;
		}
	}
	else
	{
		printf("\nTSU Error: In getMassCollision function bodyId invalid\n");
		exit(0);
	}
	return(mass);
}

float3 getCenterOfMassCollision(int scope)
{
	float totalMass;
	float3 centerOfMass;
	centerOfMass.x = 0.0;
	centerOfMass.y = 0.0;
	centerOfMass.z = 0.0;
	
	if(scope == 0) // Entire System
	{
		for(int i = 0; i < N; i++)
		{
			if(i < NFe)
			{
		    	centerOfMass.x += Pos[i].x*MassFe;
				centerOfMass.y += Pos[i].y*MassFe;
				centerOfMass.z += Pos[i].z*MassFe;
			}
			else
			{
		    	centerOfMass.x += Pos[i].x*MassSi;
				centerOfMass.y += Pos[i].y*MassSi;
				centerOfMass.z += Pos[i].z*MassSi;
			}
		}
		totalMass = getMassCollision(0);
		centerOfMass.x /= totalMass;
		centerOfMass.y /= totalMass;
		centerOfMass.z /= totalMass;
	}
	else if(scope == 1) // Earth-Moon System
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe)
			{
		    		centerOfMass.x += Pos[EarthIndex[i]].x*MassFe;
				centerOfMass.y += Pos[EarthIndex[i]].y*MassFe;
				centerOfMass.z += Pos[EarthIndex[i]].z*MassFe;
			}
			else
			{
		    		centerOfMass.x += Pos[EarthIndex[i]].x*MassSi;
				centerOfMass.y += Pos[EarthIndex[i]].y*MassSi;
				centerOfMass.z += Pos[EarthIndex[i]].z*MassSi;
			}
		}
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe)
			{
		    		centerOfMass.x += Pos[MoonIndex[i]].x*MassFe;
				centerOfMass.y += Pos[MoonIndex[i]].y*MassFe;
				centerOfMass.z += Pos[MoonIndex[i]].z*MassFe;
			}
			else
			{
		    		centerOfMass.x += Pos[MoonIndex[i]].x*MassSi;
				centerOfMass.y += Pos[MoonIndex[i]].y*MassSi;
				centerOfMass.z += Pos[MoonIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(1);
		centerOfMass.x /= totalMass;
		centerOfMass.y /= totalMass;
		centerOfMass.z /= totalMass;
		
	}
	else if(scope == 2) // Earth
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe)
			{
		    		centerOfMass.x += Pos[EarthIndex[i]].x*MassFe;
				centerOfMass.y += Pos[EarthIndex[i]].y*MassFe;
				centerOfMass.z += Pos[EarthIndex[i]].z*MassFe;
			}
			else
			{
		    		centerOfMass.x += Pos[EarthIndex[i]].x*MassSi;
				centerOfMass.y += Pos[EarthIndex[i]].y*MassSi;
				centerOfMass.z += Pos[EarthIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(2);
		centerOfMass.x /= totalMass;
		centerOfMass.y /= totalMass;
		centerOfMass.z /= totalMass;
	}
	else if(scope == 3) // Moon
	{
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe)
			{
		    		centerOfMass.x += Pos[MoonIndex[i]].x*MassFe;
				centerOfMass.y += Pos[MoonIndex[i]].y*MassFe;
				centerOfMass.z += Pos[MoonIndex[i]].z*MassFe;
			}
			else
			{
		    		centerOfMass.x += Pos[MoonIndex[i]].x*MassSi;
				centerOfMass.y += Pos[MoonIndex[i]].y*MassSi;
				centerOfMass.z += Pos[MoonIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(3);
		centerOfMass.x /= totalMass;
		centerOfMass.y /= totalMass;
		centerOfMass.z /= totalMass;
	}
	else
	{
		printf("\nTSU Error: In getCenterOfMassCollision function scope invalid\n");
		exit(0);
	}
	return(centerOfMass);
}

float3 getLinearVelocityCollision(int scope)
{
	float totalMass;
	float3 linearVelocity;
	linearVelocity.x = 0.0;
	linearVelocity.y = 0.0;
	linearVelocity.z = 0.0;
	
	if(scope == 0) // entire system
	{
		for(int i = 0; i < N; i++)
		{
			if(i < NFe)
			{
		    		linearVelocity.x += Vel[i].x*MassFe;
				linearVelocity.y += Vel[i].y*MassFe;
				linearVelocity.z += Vel[i].z*MassFe;
			}
			else
			{
		    		linearVelocity.x += Vel[i].x*MassSi;
				linearVelocity.y += Vel[i].y*MassSi;
				linearVelocity.z += Vel[i].z*MassSi;
			}
		}
		totalMass = getMassCollision(0);
		linearVelocity.x /= totalMass;
		linearVelocity.y /= totalMass;
		linearVelocity.z /= totalMass;
	}	
	else if(scope == 1) // earth-moon system
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe)
			{
		    	linearVelocity.x += Vel[EarthIndex[i]].x*MassFe;
				linearVelocity.y += Vel[EarthIndex[i]].y*MassFe;
				linearVelocity.z += Vel[EarthIndex[i]].z*MassFe;
			}
			else
			{
		    	linearVelocity.x += Vel[EarthIndex[i]].x*MassSi;
				linearVelocity.y += Vel[EarthIndex[i]].y*MassSi;
				linearVelocity.z += Vel[EarthIndex[i]].z*MassSi;
			}
		}
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe)
			{
		    	linearVelocity.x += Vel[MoonIndex[i]].x*MassFe;
				linearVelocity.y += Vel[MoonIndex[i]].y*MassFe;
				linearVelocity.z += Vel[MoonIndex[i]].z*MassFe;
			}
			else
			{
		    	linearVelocity.x += Vel[MoonIndex[i]].x*MassSi;
				linearVelocity.y += Vel[MoonIndex[i]].y*MassSi;
				linearVelocity.z += Vel[MoonIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(1);
		linearVelocity.x /= totalMass;
		linearVelocity.y /= totalMass;
		linearVelocity.z /= totalMass;
	}
	else if(scope == 2) //earth
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe)
			{
		    	linearVelocity.x += Vel[EarthIndex[i]].x*MassFe;
				linearVelocity.y += Vel[EarthIndex[i]].y*MassFe;
				linearVelocity.z += Vel[EarthIndex[i]].z*MassFe;
			}
			else
			{
		    	linearVelocity.x += Vel[EarthIndex[i]].x*MassSi;
				linearVelocity.y += Vel[EarthIndex[i]].y*MassSi;
				linearVelocity.z += Vel[EarthIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(2);
		linearVelocity.x /= totalMass;
		linearVelocity.y /= totalMass;
		linearVelocity.z /= totalMass;
	}
	else if(scope == 3) //moon
	{
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe)
			{
		    	linearVelocity.x += Vel[MoonIndex[i]].x*MassFe;
				linearVelocity.y += Vel[MoonIndex[i]].y*MassFe;
				linearVelocity.z += Vel[MoonIndex[i]].z*MassFe;
			}
			else
			{
		    	linearVelocity.x += Vel[MoonIndex[i]].x*MassSi;
				linearVelocity.y += Vel[MoonIndex[i]].y*MassSi;
				linearVelocity.z += Vel[MoonIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(3);
		linearVelocity.x /= totalMass;
		linearVelocity.y /= totalMass;
		linearVelocity.z /= totalMass;
	}
	else
	{
		printf("\nTSU Error: in getlinearVelocityEarthMoonSystem function scope invalid\n");
		exit(0);
	}
	return(linearVelocity);
}

float3 getAngularMomentumCollision(int scope)
{
	float3 centerOfMass, linearVelocity, angularMomentum;
	float3 r;
	float3 v;
	angularMomentum.x = 0.0;
	angularMomentum.y = 0.0;
	angularMomentum.z = 0.0;
	
	if(scope == 0) //Entire system
	{
		centerOfMass = getCenterOfMassCollision(0);
		linearVelocity = getLinearVelocityCollision(0);
		for(int i = 0; i < N; i++)
		{
			r.x = Pos[i].x - centerOfMass.x;
			r.y = Pos[i].y - centerOfMass.y;
			r.z = Pos[i].z - centerOfMass.z;
		
			v.x = Vel[i].x - linearVelocity.x;
			v.y = Vel[i].y - linearVelocity.y;
			v.z = Vel[i].z - linearVelocity.z;
			if(i < NFe)
			{
		    		angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassFe;
				angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassFe;
				angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassFe;
			}
			else
			{
				angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassSi;
				angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassSi;
				angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassSi;
			}
		}
	}
	else if(scope == 1) //Earth-Moon system
	{
		centerOfMass = getCenterOfMassCollision(1);
		linearVelocity = getLinearVelocityCollision(1);
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			r.x = Pos[EarthIndex[i]].x - centerOfMass.x;
			r.y = Pos[EarthIndex[i]].y - centerOfMass.y;
			r.z = Pos[EarthIndex[i]].z - centerOfMass.z;
		
			v.x = Vel[EarthIndex[i]].x - linearVelocity.x;
			v.y = Vel[EarthIndex[i]].y - linearVelocity.y;
			v.z = Vel[EarthIndex[i]].z - linearVelocity.z;
			if(EarthIndex[i] < NFe)
			{
		    		angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassFe;
				angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassFe;
				angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassFe;
			}
			else
			{
				angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassSi;
				angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassSi;
				angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassSi;
			}
		}
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			r.x = Pos[MoonIndex[i]].x - centerOfMass.x;
			r.y = Pos[MoonIndex[i]].y - centerOfMass.y;
			r.z = Pos[MoonIndex[i]].z - centerOfMass.z;
		
			v.x = Vel[MoonIndex[i]].x - linearVelocity.x;
			v.y = Vel[MoonIndex[i]].y - linearVelocity.y;
			v.z = Vel[MoonIndex[i]].z - linearVelocity.z;
			if(MoonIndex[i] < NFe)
			{
		    		angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassFe;
				angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassFe;
				angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassFe;
			}
			else
			{
				angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassSi;
				angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassSi;
				angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassSi;
			}
		}
	}
	else if(scope == 2) //Earth
	{
		centerOfMass = getCenterOfMassCollision(2);
		linearVelocity = getLinearVelocityCollision(2);
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			r.x = Pos[EarthIndex[i]].x - centerOfMass.x;
			r.y = Pos[EarthIndex[i]].y - centerOfMass.y;
			r.z = Pos[EarthIndex[i]].z - centerOfMass.z;
		
			v.x = Vel[EarthIndex[i]].x - linearVelocity.x;
			v.y = Vel[EarthIndex[i]].y - linearVelocity.y;
			v.z = Vel[EarthIndex[i]].z - linearVelocity.z;
			if(EarthIndex[i] < NFe)
			{
		    		angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassFe;
				angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassFe;
				angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassFe;
			}
			else
			{
				angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassSi;
				angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassSi;
				angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassSi;
			}
		}
	}
	else if(scope == 3) //Moon
	{
		centerOfMass = getCenterOfMassCollision(3);
		linearVelocity = getLinearVelocityCollision(3);
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			r.x = Pos[MoonIndex[i]].x - centerOfMass.x;
			r.y = Pos[MoonIndex[i]].y - centerOfMass.y;
			r.z = Pos[MoonIndex[i]].z - centerOfMass.z;
		
			v.x = Vel[MoonIndex[i]].x - linearVelocity.x;
			v.y = Vel[MoonIndex[i]].y - linearVelocity.y;
			v.z = Vel[MoonIndex[i]].z - linearVelocity.z;
			if(MoonIndex[i] < NFe)
			{
		    		angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassFe;
				angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassFe;
				angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassFe;
			}
			else
			{
				angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassSi;
				angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassSi;
				angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassSi;
			}
		}
	}
	else
	{
		printf("\nTSU Error: in getAngularMomentumCollision function scope invalid\n");
		exit(0);
	}
	return(angularMomentum);
}

void printContinueStatsToScreen(double time)
{	
	double timeConverter = UnitTime;
	double lengthConverter = UnitLength;
	double massConverter = UnitMass; 
	//double velocityConverter = UnitLength/UnitTime; 
	double momentumConverter = UnitMass*UnitLength*UnitLength/UnitTime;
	
	float3 r, v;
	double d, mass, mag, size, angle, x, y, z;
	
	float massEarth;
	float3 centerOfMassEarth;
	float3 linearVelocityEarth;
	
	float3 centerOfMassEarthMoonMaterial;
	float3 averageVelocityEarthMoonMaterial;
	
	int earthMaterialFeCountBody1 = 0;
	int earthMaterialFeCountBody2 = 0;
	int earthMaterialSiCountBody1 = 0;
	int earthMaterialSiCountBody2 = 0;
	float earthMaterialMass = 0.0;
	
	int moonMaterialFeCountBody1 = 0;
	int moonMaterialFeCountBody2 = 0;
	int moonMaterialSiCountBody1 = 0;
	int moonMaterialSiCountBody2 = 0;
	float moonMaterialMass = 0.0;
	
	int escapeMaterialFeCountBody1 = 0;
	int escapeMaterialFeCountBody2 = 0;
	int escapeMaterialSiCountBody1 = 0;
	int escapeMaterialSiCountBody2 = 0;
	float escapeMaterialMass = 0.0;
	
	int unusedMaterialFeCountBody1 = 0;
	int unusedMaterialFeCountBody2 = 0;
	int unusedMaterialSiCountBody1 = 0;
	int unusedMaterialSiCountBody2 = 0;
	float unusedMaterialMass = 0.0;
	
	float3 angularMomentumHolder;
	float3 angularMomentumEarthMoonMaterial;
	float3 angularMomentumEarthMaterial;
	float3 angularMomentumMoonMaterial;
	
	//Finding radius of what the current Earth is
	findEarthAndMoon();
	centerOfMassEarth = getCenterOfMassCollision(2);
	massEarth = getMassCollision(2);
	float radiusOfEarth = 0.0;
	for(int i = 0; i < NumberOfEarthElements; i++)
	{
		r.x = Pos[EarthIndex[i]].x - centerOfMassEarth.x;
		r.y = Pos[EarthIndex[i]].y - centerOfMassEarth.y;
		r.z = Pos[EarthIndex[i]].z - centerOfMassEarth.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(d > radiusOfEarth) radiusOfEarth = d;
	}
	
	// Finding Roche limit and setting sphere to create Earth and sphere to create the Moon 
	float densityEarth = massEarth/((Pi*4.0/3.0)*radiusOfEarth*radiusOfEarth*radiusOfEarth);
	float densitySi = MassSi/((Pi*4.0/3.0)*(Diameter/2.0)*(Diameter/2.0)*(Diameter/2.0));
	float rocheLimit = 2.44*radiusOfEarth*pow((densityEarth/densitySi),1.0/3.0);
	float radiusEarthMaterial = rocheLimit;
	float radiusMoonMaterial  = NUMBEROFEARTHRADIFORMOONMATERIAL*radiusOfEarth;
	
	// Finding mass of Earth material, Moon Material
	// Finding the center of mass and average velocity of the material we estimating will make the Earth-Moon system 
	// Finding Moon mix and Earth mix
	earthMaterialMass = 0.0;
	moonMaterialMass = 0.0;
	
	centerOfMassEarthMoonMaterial.x = 0.0;
	centerOfMassEarthMoonMaterial.y = 0.0;
	centerOfMassEarthMoonMaterial.z = 0.0;
	
	averageVelocityEarthMoonMaterial.x = 0.0;
	averageVelocityEarthMoonMaterial.y = 0.0;
	averageVelocityEarthMoonMaterial.z = 0.0;
	
	for(int i = 0; i < N; i++)
	{
		
		r.x = Pos[i].x - centerOfMassEarth.x;
		r.y = Pos[i].y - centerOfMassEarth.y;
		r.z = Pos[i].z - centerOfMassEarth.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(d < radiusEarthMaterial)
		{
			if(i < NFe) 	mass = MassFe;	
			else 		mass = MassSi;
			
			earthMaterialMass += mass;
			
			centerOfMassEarthMoonMaterial.x += mass*Pos->x;
			centerOfMassEarthMoonMaterial.y += mass*Pos->y;
			centerOfMassEarthMoonMaterial.z += mass*Pos->z;
			
			averageVelocityEarthMoonMaterial.x += mass*Vel->x;
			averageVelocityEarthMoonMaterial.y += mass*Vel->y;
			averageVelocityEarthMoonMaterial.z += mass*Vel->z;
			
			if(i < NFe1) 				earthMaterialFeCountBody1++;
			else if(i < NFe1 + NFe2) 		earthMaterialFeCountBody2++;
			else if(i < NFe1 + NFe2 + NSi1) 	earthMaterialSiCountBody1++;
			else					earthMaterialSiCountBody2++;
		}
		else if(d < radiusMoonMaterial)
		{
			if(i < NFe) 	mass = MassFe;
			else 		mass = MassSi;
			
			moonMaterialMass += mass;
			
			centerOfMassEarthMoonMaterial.x += mass*Pos->x;
			centerOfMassEarthMoonMaterial.y += mass*Pos->y;
			centerOfMassEarthMoonMaterial.z += mass*Pos->z;
			
			averageVelocityEarthMoonMaterial.x += mass*Vel->x;
			averageVelocityEarthMoonMaterial.y += mass*Vel->y;
			averageVelocityEarthMoonMaterial.z += mass*Vel->z;
			
			if(i < NFe1) 				moonMaterialFeCountBody1++;
			else if(i < NFe1 + NFe2) 		moonMaterialFeCountBody2++;
			else if(i < NFe1 + NFe2 + NSi1) 	moonMaterialSiCountBody1++;
			else					moonMaterialSiCountBody2++;
			
		}
	}
	centerOfMassEarthMoonMaterial.x /= (earthMaterialMass + moonMaterialMass);
	centerOfMassEarthMoonMaterial.y /= (earthMaterialMass + moonMaterialMass);
	centerOfMassEarthMoonMaterial.z /= (earthMaterialMass + moonMaterialMass);
	
	averageVelocityEarthMoonMaterial.x /= (earthMaterialMass + moonMaterialMass);
	averageVelocityEarthMoonMaterial.y /= (earthMaterialMass + moonMaterialMass);
	averageVelocityEarthMoonMaterial.z /= (earthMaterialMass + moonMaterialMass);
	
	// Getting a rough estimate of how much of the extra material has escape velocity from what we 
	// considering will make the Earth-Moon system
	float velocity;
	float escapeVelocity;
	escapeMaterialMass = 0.0;
	unusedMaterialMass = 0.0;
	for(int i = 0; i < N; i++)
	{
		r.x = Pos[i].x - centerOfMassEarth.x;
		r.y = Pos[i].y - centerOfMassEarth.y;
		r.z = Pos[i].z - centerOfMassEarth.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(radiusMoonMaterial <= d)
		{
			r.x = Pos[i].x - centerOfMassEarthMoonMaterial.x;
			r.y = Pos[i].y - centerOfMassEarthMoonMaterial.y;
			r.z = Pos[i].z - centerOfMassEarthMoonMaterial.z;
			d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
			
			v.x = Vel[i].x - averageVelocityEarthMoonMaterial.x;
			v.y = Vel[i].y - averageVelocityEarthMoonMaterial.y;
			v.z = Vel[i].z - averageVelocityEarthMoonMaterial.z;
			velocity = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
		
			escapeVelocity = sqrt(2.0*Gravity*(earthMaterialMass + moonMaterialMass)/d);
			
			if(velocity >= escapeVelocity)
			{
				if(i < NFe) 	mass = MassFe;
				else 		mass = MassSi;
			
				escapeMaterialMass += mass;
			
			 	if(i < NFe1) 				escapeMaterialFeCountBody1++;
				else if(i < NFe1 + NFe2) 		escapeMaterialFeCountBody2++;
				else if(i < NFe1 + NFe2 + NSi1) 	escapeMaterialSiCountBody1++;
				else					escapeMaterialSiCountBody2++;
			}
			else
			{
				if(i < NFe) 	mass = MassFe;
				else 		mass = MassSi;
			
				unusedMaterialMass += mass;
				if(i < NFe1) 				unusedMaterialFeCountBody1++;
				else if(i < NFe1 + NFe2) 		unusedMaterialFeCountBody2++;
				else if(i < NFe1 + NFe2 + NSi1) 	unusedMaterialSiCountBody1++;
				else					unusedMaterialSiCountBody2++;
			}
		}
	}
	
	// Finding the angular momentum of the Earth-Moon material
	// Finding the angular momentum of the Earth material
	// Finding the angular momentum of the Moon material
	linearVelocityEarth = getLinearVelocityCollision(2);
	
	angularMomentumEarthMoonMaterial.x = 0.0;
	angularMomentumEarthMoonMaterial.y = 0.0;
	angularMomentumEarthMoonMaterial.z = 0.0;
	
	angularMomentumEarthMaterial.x = 0.0;
	angularMomentumEarthMaterial.y = 0.0;
	angularMomentumEarthMaterial.z = 0.0;
	
	angularMomentumMoonMaterial.x = 0.0;
	angularMomentumMoonMaterial.y = 0.0;
	angularMomentumMoonMaterial.z = 0.0;
	
	for(int i = 0; i < N; i++)
	{
		r.x = Pos[i].x - centerOfMassEarth.x;
		r.y = Pos[i].y - centerOfMassEarth.y;
		r.z = Pos[i].z - centerOfMassEarth.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(d < radiusMoonMaterial)
		{
			v.x = Vel[i].x - linearVelocityEarth.x;
			v.y = Vel[i].y - linearVelocityEarth.y;
			v.z = Vel[i].z - linearVelocityEarth.z;
			if(i < NFe)
			{
		    		angularMomentumHolder.x =  (r.y*v.z - r.z*v.y)*MassFe;
				angularMomentumHolder.y = -(r.x*v.z - r.z*v.x)*MassFe;
				angularMomentumHolder.z =  (r.x*v.y - r.y*v.x)*MassFe;
			}
			else
			{
				angularMomentumHolder.x =  (r.y*v.z - r.z*v.y)*MassSi;
				angularMomentumHolder.y = -(r.x*v.z - r.z*v.x)*MassSi;
				angularMomentumHolder.z =  (r.x*v.y - r.y*v.x)*MassSi;
			}
			
			angularMomentumEarthMoonMaterial.x += angularMomentumHolder.x;
			angularMomentumEarthMoonMaterial.y += angularMomentumHolder.y;
			angularMomentumEarthMoonMaterial.z += angularMomentumHolder.z;
			
			if(d < radiusEarthMaterial)
			{
				angularMomentumEarthMaterial.x +=  angularMomentumHolder.x;
				angularMomentumEarthMaterial.y +=  angularMomentumHolder.y;
				angularMomentumEarthMaterial.z +=  angularMomentumHolder.z;
			}
			else
			{
				angularMomentumMoonMaterial.x +=  angularMomentumHolder.x;
				angularMomentumMoonMaterial.y +=  angularMomentumHolder.y;
				angularMomentumMoonMaterial.z +=  angularMomentumHolder.z;
			}
		}
	}
	
	printf("\n\n\n*************************************************************************\n");
	printf("\nThe following are the three stats to feed to the search program\n");
	
	x = angularMomentumEarthMoonMaterial.x*momentumConverter;
	y = angularMomentumEarthMoonMaterial.y*momentumConverter;
	z = angularMomentumEarthMoonMaterial.z*momentumConverter;
	mag = sqrt(x*x + y*y + z*z);
	printf("\nAngular momentum of the Earth-Moon system = %e", mag);
	printf("\nRatio Earth mass to Moon mass = %f", earthMaterialMass/moonMaterialMass);
	printf("\nMoon compotition ratio  = %f", (float)(moonMaterialFeCountBody1 + moonMaterialSiCountBody1)/(float)(moonMaterialFeCountBody2 + moonMaterialSiCountBody2));
	
	printf("\n\n\n*************************************************************************\n");
	printf("\nThe following are all the continuation stats of the run when time = %f hours\n", time*timeConverter/3600.0);
	printf("\nDistance is measured in Kilometers");
	printf("\nMass is measured in Kilograms");
	printf("\nTime is measured in seconds");
	printf("\nVelocity is measured in Kilometers/second");
	printf("\nAngular momentun is measured in Kilograms*Kilometers*Kilometers/seconds\n");
	
	printf("\nThe radius of Earth 		= %f", radiusOfEarth*lengthConverter);
	printf("\nRoche limit 			= %f", rocheLimit*lengthConverter);
	printf("\nRoche limit/radius of Earth 	= %f \n", rocheLimit/radiusOfEarth);
	
	x = angularMomentumEarthMoonMaterial.x*momentumConverter;
	y = angularMomentumEarthMoonMaterial.y*momentumConverter;
	z = angularMomentumEarthMoonMaterial.z*momentumConverter;
	printf("\nAngular momentum of the Earth-Moon material			 = (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	printf("\nMagnitude of the angular momentum of the Earth-Moon material 	 = %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	printf("\nAngle off ecliptic plane of the Earth-Moon's material rotation	 = %f\n", 90.0 - angle*180.0/Pi);
	
	x = angularMomentumEarthMaterial.x*momentumConverter;
	y = angularMomentumEarthMaterial.y*momentumConverter;
	z = angularMomentumEarthMaterial.z*momentumConverter;
	printf("\nAngular momentum of the Earth material				= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	printf("\nMagnitude of the angular momentum of the Earth material 	= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	printf("\nAngle off ecliptic plane of the Earth's material rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	x = angularMomentumMoonMaterial.x*momentumConverter;
	y = angularMomentumMoonMaterial.y*momentumConverter;
	z = angularMomentumMoonMaterial.z*momentumConverter;
	printf("\nAngular momentum of the Moon material			   	= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	printf("\nMagnitude of the angular momentum of the Moon material   	= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	printf("\nAngle off ecliptic plane of the Moon's material rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	printf("\nThe mass of Earth material		= %e", earthMaterialMass*massConverter);
	printf("\nThe Earth material count Fe body 1	= %d", earthMaterialFeCountBody1);
	printf("\nThe Earth material count Fe body 2	= %d", earthMaterialFeCountBody2);
	printf("\nThe Earth material count Si body 1	= %d", earthMaterialSiCountBody1);
	printf("\nThe Earth material count Si body 2	= %d", earthMaterialSiCountBody2);
	printf("\nThe Earth material Body1/Body2 ratio	= %f\n", (float)(earthMaterialFeCountBody1 + earthMaterialSiCountBody1)/(float)(earthMaterialFeCountBody2 + earthMaterialSiCountBody2));
	
	printf("\nThe mass of Moon material		= %e", moonMaterialMass*massConverter);
	printf("\nThe Moon material count Fe body 1	= %d", moonMaterialFeCountBody1);
	printf("\nThe Moon material count Fe body 2	= %d", moonMaterialFeCountBody2);
	printf("\nThe Moon material count Si body 1	= %d", moonMaterialSiCountBody1);
	printf("\nThe Moon material count Si body 2	= %d", moonMaterialSiCountBody2);
	printf("\nThe Moon material Body1/Body2 ratio	= %f\n", (float)(moonMaterialFeCountBody1 + moonMaterialSiCountBody1)/(float)(moonMaterialFeCountBody2 + moonMaterialSiCountBody2));
	
	printf("\nThe mass of escape material		= %e", escapeMaterialMass*massConverter);
	printf("\nThe escape material count Fe body 1	= %d", escapeMaterialFeCountBody1);
	printf("\nThe escape material count Fe body 2	= %d", escapeMaterialFeCountBody2);
	printf("\nThe escape material count Si body 1	= %d", escapeMaterialSiCountBody1);
	printf("\nThe escape material count Si body 2	= %d", escapeMaterialSiCountBody2);
	printf("\nThe escape material Body1/Body2 ratio	= %f\n", (float)(escapeMaterialFeCountBody1 + escapeMaterialSiCountBody1)/(float)(escapeMaterialFeCountBody2 + escapeMaterialSiCountBody2));
	
	printf("\nThe mass of unused material		= %e", unusedMaterialMass*massConverter);
	printf("\nThe unused material count Fe body 1	= %d", unusedMaterialFeCountBody1);
	printf("\nThe unused material count Fe body 2	= %d", unusedMaterialFeCountBody2);
	printf("\nThe unused material count Si body 1	= %d", unusedMaterialSiCountBody1);
	printf("\nThe unused material count Si body 2	= %d", unusedMaterialSiCountBody2);
	printf("\nThe unused material Body1/Body2 ratio	= %f\n", (float)(unusedMaterialFeCountBody1 + unusedMaterialSiCountBody1)/(float)(unusedMaterialFeCountBody2 + unusedMaterialSiCountBody2));
	
	printf("\n*************************************************************************\n\n\n");
}

void printContinueStatsToFile(double time)
{	
	double timeConverter = UnitTime;
	double lengthConverter = UnitLength;
	double massConverter = UnitMass; 
	//double velocityConverter = UnitLength/UnitTime; 
	double momentumConverter = UnitMass*UnitLength*UnitLength/UnitTime;
	
	float3 r, v;
	double d, mass, mag, size, angle, x, y, z;
	
	float massEarth;
	float3 centerOfMassEarth;
	float3 linearVelocityEarth;
	
	float3 centerOfMassEarthMoonMaterial;
	float3 averageVelocityEarthMoonMaterial;
	
	int earthMaterialFeCountBody1 = 0;
	int earthMaterialFeCountBody2 = 0;
	int earthMaterialSiCountBody1 = 0;
	int earthMaterialSiCountBody2 = 0;
	float earthMaterialMass = 0.0;
	
	int moonMaterialFeCountBody1 = 0;
	int moonMaterialFeCountBody2 = 0;
	int moonMaterialSiCountBody1 = 0;
	int moonMaterialSiCountBody2 = 0;
	float moonMaterialMass = 0.0;
	
	int escapeMaterialFeCountBody1 = 0;
	int escapeMaterialFeCountBody2 = 0;
	int escapeMaterialSiCountBody1 = 0;
	int escapeMaterialSiCountBody2 = 0;
	float escapeMaterialMass = 0.0;
	
	int unusedMaterialFeCountBody1 = 0;
	int unusedMaterialFeCountBody2 = 0;
	int unusedMaterialSiCountBody1 = 0;
	int unusedMaterialSiCountBody2 = 0;
	float unusedMaterialMass = 0.0;
	
	float3 angularMomentumEarthMoonMaterial;
	float3 angularMomentumEarthMaterial;
	float3 angularMomentumMoonMaterial;
	
	//Finding radius of what the current Earth is
	findEarthAndMoon();
	centerOfMassEarth = getCenterOfMassCollision(2);
	massEarth = getMassCollision(2);
	float radiusOfEarth = 0.0;
	for(int i = 0; i < NumberOfEarthElements; i++)
	{
		r.x = Pos[EarthIndex[i]].x - centerOfMassEarth.x;
		r.y = Pos[EarthIndex[i]].y - centerOfMassEarth.y;
		r.z = Pos[EarthIndex[i]].z - centerOfMassEarth.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(d > radiusOfEarth) radiusOfEarth = d;
	}
	
	// Finding Roche limit and setting sphere to create Earth and sphere to create the Moon 
	float densityEarth = massEarth/((Pi*4.0/3.0)*radiusOfEarth*radiusOfEarth*radiusOfEarth);
	float densitySi = MassSi/((Pi*4.0/3.0)*(Diameter/2.0)*(Diameter/2.0)*(Diameter/2.0));
	float rocheLimit = 2.44*radiusOfEarth*pow((densityEarth/densitySi),1.0/3.0);
	float radiusEarthMaterial = rocheLimit;
	float radiusMoonMaterial  = NUMBEROFEARTHRADIFORMOONMATERIAL*radiusOfEarth;
	
	// Finding mass of Earth material, Moon Material
	// Finding the center of mass and average velocity of the material we estimating will make the Earth-Moon system 
	// Finding Moon mix and Earth mix
	earthMaterialMass = 0.0;
	moonMaterialMass = 0.0;
	
	centerOfMassEarthMoonMaterial.x = 0.0;
	centerOfMassEarthMoonMaterial.y = 0.0;
	centerOfMassEarthMoonMaterial.z = 0.0;
	
	averageVelocityEarthMoonMaterial.x = 0.0;
	averageVelocityEarthMoonMaterial.y = 0.0;
	averageVelocityEarthMoonMaterial.z = 0.0;
	
	for(int i = 0; i < N; i++)
	{
		
		r.x = Pos[i].x - centerOfMassEarth.x;
		r.y = Pos[i].y - centerOfMassEarth.y;
		r.z = Pos[i].z - centerOfMassEarth.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(d < radiusEarthMaterial)
		{
			if(i < NFe) 	mass = MassFe;	
			else 		mass = MassSi;
			
			earthMaterialMass += mass;
			
			centerOfMassEarthMoonMaterial.x += mass*Pos->x;
			centerOfMassEarthMoonMaterial.y += mass*Pos->y;
			centerOfMassEarthMoonMaterial.z += mass*Pos->z;
			
			averageVelocityEarthMoonMaterial.x += mass*Vel->x;
			averageVelocityEarthMoonMaterial.y += mass*Vel->y;
			averageVelocityEarthMoonMaterial.z += mass*Vel->z;
			
			if(i < NFe1) 				earthMaterialFeCountBody1++;
			else if(i < NFe1 + NFe2) 		earthMaterialFeCountBody2++;
			else if(i < NFe1 + NFe2 + NSi1) 	earthMaterialSiCountBody1++;
			else					earthMaterialSiCountBody2++;
		}
		else if(d < radiusMoonMaterial)
		{
			if(i < NFe) 	mass = MassFe;
			else 		mass = MassSi;
			
			moonMaterialMass += mass;
			
			centerOfMassEarthMoonMaterial.x += mass*Pos->x;
			centerOfMassEarthMoonMaterial.y += mass*Pos->y;
			centerOfMassEarthMoonMaterial.z += mass*Pos->z;
			
			averageVelocityEarthMoonMaterial.x += mass*Vel->x;
			averageVelocityEarthMoonMaterial.y += mass*Vel->y;
			averageVelocityEarthMoonMaterial.z += mass*Vel->z;
			
			if(i < NFe1) 				moonMaterialFeCountBody1++;
			else if(i < NFe1 + NFe2) 		moonMaterialFeCountBody2++;
			else if(i < NFe1 + NFe2 + NSi1) 	moonMaterialSiCountBody1++;
			else					moonMaterialSiCountBody2++;
			
		}
	}
	centerOfMassEarthMoonMaterial.x /= (earthMaterialMass + moonMaterialMass);
	centerOfMassEarthMoonMaterial.y /= (earthMaterialMass + moonMaterialMass);
	centerOfMassEarthMoonMaterial.z /= (earthMaterialMass + moonMaterialMass);
	
	averageVelocityEarthMoonMaterial.x /= (earthMaterialMass + moonMaterialMass);
	averageVelocityEarthMoonMaterial.y /= (earthMaterialMass + moonMaterialMass);
	averageVelocityEarthMoonMaterial.z /= (earthMaterialMass + moonMaterialMass);
	
	// Getting a rough estimate of how much of the extra material has escape velocity from what we 
	// considering will make the Earth-Moon system
	float velocity;
	float escapeVelocity;
	escapeMaterialMass = 0.0;
	unusedMaterialMass = 0.0;
	for(int i = 0; i < N; i++)
	{
		r.x = Pos[i].x - centerOfMassEarth.x;
		r.y = Pos[i].y - centerOfMassEarth.y;
		r.z = Pos[i].z - centerOfMassEarth.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(radiusMoonMaterial <= d)
		{
			r.x = Pos[i].x - centerOfMassEarthMoonMaterial.x;
			r.y = Pos[i].y - centerOfMassEarthMoonMaterial.y;
			r.z = Pos[i].z - centerOfMassEarthMoonMaterial.z;
			d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
			
			v.x = Vel[i].x - averageVelocityEarthMoonMaterial.x;
			v.y = Vel[i].y - averageVelocityEarthMoonMaterial.y;
			v.z = Vel[i].z - averageVelocityEarthMoonMaterial.z;
			velocity = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
		
			escapeVelocity = sqrt(2.0*Gravity*(earthMaterialMass + moonMaterialMass)/d);
			
			if(velocity >= escapeVelocity)
			{
				if(i < NFe) 	mass = MassFe;
				else 		mass = MassSi;
			
				escapeMaterialMass += mass;
			
			 	if(i < NFe1) 				escapeMaterialFeCountBody1++;
				else if(i < NFe1 + NFe2) 		escapeMaterialFeCountBody2++;
				else if(i < NFe1 + NFe2 + NSi1) 	escapeMaterialSiCountBody1++;
				else					escapeMaterialSiCountBody2++;
			}
			else
			{
				if(i < NFe) 	mass = MassFe;
				else 		mass = MassSi;
			
				unusedMaterialMass += mass;
				if(i < NFe1) 				unusedMaterialFeCountBody1++;
				else if(i < NFe1 + NFe2) 		unusedMaterialFeCountBody2++;
				else if(i < NFe1 + NFe2 + NSi1) 	unusedMaterialSiCountBody1++;
				else					unusedMaterialSiCountBody2++;
			}
		}
	}
	
	// Finding the angular momentum of the Earth-Moon material
	// Finding the angular momentum of the Earth material
	// Finding the angular momentum of the Moon material
	linearVelocityEarth = getLinearVelocityCollision(2);
	
	angularMomentumEarthMoonMaterial.x = 0.0;
	angularMomentumEarthMoonMaterial.y = 0.0;
	angularMomentumEarthMoonMaterial.z = 0.0;
	
	angularMomentumEarthMaterial.x = 0.0;
	angularMomentumEarthMaterial.y = 0.0;
	angularMomentumEarthMaterial.z = 0.0;
	
	angularMomentumMoonMaterial.x = 0.0;
	angularMomentumMoonMaterial.y = 0.0;
	angularMomentumMoonMaterial.z = 0.0;
	
	for(int i = 0; i < N; i++)
	{
		r.x = Pos[i].x - centerOfMassEarth.x;
		r.y = Pos[i].y - centerOfMassEarth.y;
		r.z = Pos[i].z - centerOfMassEarth.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(d < radiusMoonMaterial)
		{
			v.x = Vel[i].x - linearVelocityEarth.x;
			v.y = Vel[i].y - linearVelocityEarth.y;
			v.z = Vel[i].z - linearVelocityEarth.z;
			if(i < NFe)
			{
		    		angularMomentumEarthMoonMaterial.x +=  (r.y*v.z - r.z*v.y)*MassFe;
				angularMomentumEarthMoonMaterial.y += -(r.x*v.z - r.z*v.x)*MassFe;
				angularMomentumEarthMoonMaterial.z +=  (r.x*v.y - r.y*v.x)*MassFe;
			}
			else
			{
				angularMomentumEarthMoonMaterial.x +=  (r.y*v.z - r.z*v.y)*MassSi;
				angularMomentumEarthMoonMaterial.y += -(r.x*v.z - r.z*v.x)*MassSi;
				angularMomentumEarthMoonMaterial.z +=  (r.x*v.y - r.y*v.x)*MassSi;
			}
			
			if(radiusEarthMaterial < d)
			{
				angularMomentumEarthMaterial.x +=  angularMomentumEarthMoonMaterial.x;
				angularMomentumEarthMaterial.y +=  angularMomentumEarthMoonMaterial.y;
				angularMomentumEarthMaterial.z +=  angularMomentumEarthMoonMaterial.z;
			}
			else
			{
				angularMomentumMoonMaterial.x +=  angularMomentumEarthMoonMaterial.x;
				angularMomentumMoonMaterial.y +=  angularMomentumEarthMoonMaterial.y;
				angularMomentumMoonMaterial.z +=  angularMomentumEarthMoonMaterial.z;
			}
		}
	}
	
	fprintf(ContinueRunStatsFile, "\n\n\n*************************************************************************\n");
	fprintf(ContinueRunStatsFile, "\nThe following are the three stats to feed to the search program\n");
	
	x = angularMomentumEarthMoonMaterial.x*momentumConverter;
	y = angularMomentumEarthMoonMaterial.y*momentumConverter;
	z = angularMomentumEarthMoonMaterial.z*momentumConverter;
	mag = sqrt(x*x + y*y + z*z);
	fprintf(ContinueRunStatsFile, "\nAngular momentum of the Earth-Moon system = %e", mag);
	fprintf(ContinueRunStatsFile, "\nRatio Earth mass to Moon mass = %f", earthMaterialMass/moonMaterialMass);
	fprintf(ContinueRunStatsFile, "\nMoon compotition ratio  = %f", (float)(moonMaterialFeCountBody1 + moonMaterialSiCountBody1)/(float)(moonMaterialFeCountBody2 + moonMaterialSiCountBody2));
	
	fprintf(ContinueRunStatsFile, "\n\n\n*************************************************************************\n");
	fprintf(ContinueRunStatsFile, "\nThe following are all the continuation stats of the run when time = %f hours\n", time*timeConverter/3600.0);
	fprintf(ContinueRunStatsFile, "\nDistance is measured in Kilometers");
	fprintf(ContinueRunStatsFile, "\nMass is measured in Kilograms");
	fprintf(ContinueRunStatsFile, "\nTime is measured in seconds");
	fprintf(ContinueRunStatsFile, "\nVelocity is measured in Kilometers/second");
	fprintf(ContinueRunStatsFile, "\nAngular momentun is measured in Kilograms*Kilometers*Kilometers/seconds\n");
	
	fprintf(ContinueRunStatsFile, "\nThe radius of Earth 		= %f", radiusOfEarth*lengthConverter);
	fprintf(ContinueRunStatsFile, "\nRoche limit 			= %f", rocheLimit*lengthConverter);
	fprintf(ContinueRunStatsFile, "\nRoche limit/radius of Earth 	= %f \n", rocheLimit/radiusOfEarth);
	
	x = angularMomentumEarthMoonMaterial.x*momentumConverter;
	y = angularMomentumEarthMoonMaterial.y*momentumConverter;
	z = angularMomentumEarthMoonMaterial.z*momentumConverter;
	fprintf(ContinueRunStatsFile, "\nAngular momentum of the Earth-Moon material			 = (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(ContinueRunStatsFile, "\nMagnitude of the angular momentum of the Earth-Moon material 	 = %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	fprintf(ContinueRunStatsFile, "\nAngle off ecliptic plane of the Earth-Moon's material rotation	 = %f\n", 90.0 - angle*180.0/Pi);
	
	x = angularMomentumEarthMaterial.x*momentumConverter;
	y = angularMomentumEarthMaterial.y*momentumConverter;
	z = angularMomentumEarthMaterial.z*momentumConverter;
	fprintf(ContinueRunStatsFile, "\nAngular momentum of the Earth material				= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(ContinueRunStatsFile, "\nMagnitude of the angular momentum of the Earth material 	= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	fprintf(ContinueRunStatsFile, "\nAngle off ecliptic plane of the Earth's material rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	x = angularMomentumMoonMaterial.x*momentumConverter;
	y = angularMomentumMoonMaterial.y*momentumConverter;
	z = angularMomentumMoonMaterial.z*momentumConverter;
	fprintf(ContinueRunStatsFile, "\nAngular momentum of the Moon material			   	= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(ContinueRunStatsFile, "\nMagnitude of the angular momentum of the Moon material   	= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	fprintf(ContinueRunStatsFile, "\nAngle off ecliptic plane of the Moon's material rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	fprintf(ContinueRunStatsFile, "\nThe mass of Earth material		= %e", earthMaterialMass*massConverter);
	fprintf(ContinueRunStatsFile, "\nThe Earth material count Fe body 1	= %d", earthMaterialFeCountBody1);
	fprintf(ContinueRunStatsFile, "\nThe Earth material count Fe body 2	= %d", earthMaterialFeCountBody2);
	fprintf(ContinueRunStatsFile, "\nThe Earth material count Si body 1	= %d", earthMaterialSiCountBody1);
	fprintf(ContinueRunStatsFile, "\nThe Earth material count Si body 2	= %d", earthMaterialSiCountBody2);
	fprintf(ContinueRunStatsFile, "\nThe Earth material Body1/Body2 ratio	= %f\n", (float)(earthMaterialFeCountBody1 + earthMaterialSiCountBody1)/(float)(earthMaterialFeCountBody2 + earthMaterialSiCountBody2));
	
	fprintf(ContinueRunStatsFile, "\nThe mass of Moon material		= %e", moonMaterialMass*massConverter);
	fprintf(ContinueRunStatsFile, "\nThe Moon material count Fe body 1	= %d", moonMaterialFeCountBody1);
	fprintf(ContinueRunStatsFile, "\nThe Moon material count Fe body 2	= %d", moonMaterialFeCountBody2);
	fprintf(ContinueRunStatsFile, "\nThe Moon material count Si body 1	= %d", moonMaterialSiCountBody1);
	fprintf(ContinueRunStatsFile, "\nThe Moon material count Si body 2	= %d", moonMaterialSiCountBody2);
	fprintf(ContinueRunStatsFile, "\nThe Moon material Body1/Body2 ratio	= %f\n", (float)(moonMaterialFeCountBody1 + moonMaterialSiCountBody1)/(float)(moonMaterialFeCountBody2 + moonMaterialSiCountBody2));
	
	fprintf(ContinueRunStatsFile, "\nThe mass of escape material		= %e", escapeMaterialMass*massConverter);
	fprintf(ContinueRunStatsFile, "\nThe escape material count Fe body 1	= %d", escapeMaterialFeCountBody1);
	fprintf(ContinueRunStatsFile, "\nThe escape material count Fe body 2	= %d", escapeMaterialFeCountBody2);
	fprintf(ContinueRunStatsFile, "\nThe escape material count Si body 1	= %d", escapeMaterialSiCountBody1);
	fprintf(ContinueRunStatsFile, "\nThe escape material count Si body 2	= %d", escapeMaterialSiCountBody2);
	fprintf(ContinueRunStatsFile, "\nThe escape material Body1/Body2 ratio	= %f\n", (float)(escapeMaterialFeCountBody1 + escapeMaterialSiCountBody1)/(float)(escapeMaterialFeCountBody2 + escapeMaterialSiCountBody2));
	
	fprintf(ContinueRunStatsFile, "\nThe mass of unused material		= %e", unusedMaterialMass*massConverter);
	fprintf(ContinueRunStatsFile, "\nThe unused material count Fe body 1	= %d", unusedMaterialFeCountBody1);
	fprintf(ContinueRunStatsFile, "\nThe unused material count Fe body 2	= %d", unusedMaterialFeCountBody2);
	fprintf(ContinueRunStatsFile, "\nThe unused material count Si body 1	= %d", unusedMaterialSiCountBody1);
	fprintf(ContinueRunStatsFile, "\nThe unused material count Si body 2	= %d", unusedMaterialSiCountBody2);
	fprintf(ContinueRunStatsFile, "\nThe unused material Body1/Body2 ratio	= %f\n", (float)(unusedMaterialFeCountBody1 + unusedMaterialSiCountBody1)/(float)(unusedMaterialFeCountBody2 + unusedMaterialSiCountBody2));
	
	fprintf(ContinueRunStatsFile, "\n*************************************************************************\n\n\n");
}

void printCollisionStatsToScreen(double time)
{	
	double mag, size, angle, x, y, z;
	
	double timeConverter = UnitTime;
	double lengthConverter = UnitLength;
	double massConverter = UnitMass; 
	double velocityConverter = UnitLength/UnitTime; 
	double momentumConverter = UnitMass*UnitLength*UnitLength/UnitTime;
	
	findEarthAndMoon();
	int earthFeCountBody1 = 0;
	int earthFeCountBody2 = 0;
	int earthSiCountBody1 = 0;
	int earthSiCountBody2 = 0;
	int moonFeCountBody1 = 0;
	int moonFeCountBody2 = 0;
	int moonSiCountBody1 = 0;
	int moonSiCountBody2 = 0;
	
	float massUniversalSystem = getMassCollision(0);
	float massEarthMoonSystem = getMassCollision(1);
	float massEarth = getMassCollision(2);
	float massMoon = getMassCollision(3);
	
	float3 centerOfMassUniversalSystem = getCenterOfMassCollision(0);
	float3 centerOfMassEarthMoonSystem = getCenterOfMassCollision(1);
	float3 centerOfMassEarth = getCenterOfMassCollision(2);
	float3 centerOfMassMoon = getCenterOfMassCollision(3);
	
	float3 linearVelocityUniversalSystem = getLinearVelocityCollision(0);
	float3 linearVelocityEarthMoonSystem = getLinearVelocityCollision(1);
	float3 linearVelocityEarth = getLinearVelocityCollision(2);
	float3 linearVelocityMoon = getLinearVelocityCollision(3);
	
	float3 angularMomentumUniversalSystem = getAngularMomentumCollision(0);
	float3 angularMomentumEarthMoonSystem = getAngularMomentumCollision(1);
	float3 angularMomentumEarth = getAngularMomentumCollision(2);
	float3 angularMomentumMoon = getAngularMomentumCollision(3);
	
	for(int i = 0; i < NumberOfEarthElements; i++)
	{
		if(EarthIndex[i] < NFe1) 			earthFeCountBody1++;
		else if(EarthIndex[i] < NFe1 + NFe2) 		earthFeCountBody2++;
		else if(EarthIndex[i] < NFe1 + NFe2 + NSi1) 	earthSiCountBody1++;
		else 						earthSiCountBody2++;
	}
	
	for(int i = 0; i < NumberOfMoonElements; i++)
	{
		if(MoonIndex[i] < NFe1) 			moonFeCountBody1++;
		else if(MoonIndex[i] < NFe1 + NFe2) 		moonFeCountBody2++;
		else if(MoonIndex[i] < NFe1 + NFe2 + NSi1) 	moonSiCountBody1++;
		else 						moonSiCountBody2++;
	}
	
	printf("\n\n\n*************************************************************************\n\n\n");
	printf("\nThe following are the stats of the run when time = %f hours\n", time*timeConverter/3600.0);
	printf("\nDistance is measured in Kilometers");
	printf("\nMass is measured in Kilograms");
	printf("\nTime is measured in seconds");
	printf("\nVelocity is measured in Kilometers/second");
	printf("\nAngular momentun is measured in Kilograms*Kilometers*Kilometers/seconds\n");
	
	printf("\nThe mass of Earth 		= %e", massEarth*massConverter);
	printf("\nThe mass of Moon 		= %e", massMoon*massConverter);
	if(massMoon != 0.0) printf("\nThe mass ratio Earth/Moon 	= %f\n", massEarth/massMoon);
	
	printf("\nMoon iron from body 1 		= %d", moonFeCountBody1);
	printf("\nMoon silicate from body 1 	= %d", moonSiCountBody1);
	printf("\nMoon iron from body 2 		= %d", moonFeCountBody2);
	printf("\nMoon silicate from body 2 	= %d", moonSiCountBody2);
	if((moonFeCountBody2 + moonSiCountBody2) == 0)
	{
		printf("\nThe Moon is only composed of elements from body 1\n");
	}
	else if((moonFeCountBody1 + moonSiCountBody1) == 0)
	{
		printf("\nThe Moon is only composed of elements from body 2\n");
	}
	else
	{
		printf("\nMoon ratio body1/body2 		= %f\n", (float)(moonFeCountBody1 + moonSiCountBody1)/(float)(moonFeCountBody2 + moonSiCountBody2));
	}
	
	printf("\nEarth iron from body 1 		= %d", earthFeCountBody1);
	printf("\nEarth silicate from body 1 	= %d", earthSiCountBody1);
	printf("\nEarth iron from body 2 		= %d", earthFeCountBody2);
	printf("\nEarth silicate from body 2 	= %d", earthSiCountBody2);
	if((earthFeCountBody2 + earthSiCountBody2) == 0)
	{
		printf("\nThe Earth is only composed of elements from body 1\n");
	}
	else if((earthFeCountBody1 + earthSiCountBody1) == 0)
	{
		printf("\nThe Earth is only composed of elements from body 2\n");
	}
	else
	{
		printf("\nEarth ratio body1/body2 		= %f\n", (float)(earthFeCountBody1 + earthSiCountBody1)/(float)(earthFeCountBody2 + earthSiCountBody2));
	}
	
	//It is always assumed that the ecliptic plane is the xz-plane.
	x = angularMomentumEarthMoonSystem.x*momentumConverter;
	y = angularMomentumEarthMoonSystem.y*momentumConverter;
	z = angularMomentumEarthMoonSystem.z*momentumConverter;
	printf("\nAngular momentum of the Earth Moon system 		= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	printf("\nMagnitude of the angular momentum of the system 	= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	printf("\nAngle off ecliptic plane of the system's rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	x = angularMomentumEarth.x*momentumConverter;
	y = angularMomentumEarth.y*momentumConverter;
	z = angularMomentumEarth.z*momentumConverter;
	printf("\nAngular momentum of the Earth 				= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	printf("\nMagnitude of the angular momentum of the Earth 		= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	printf("\nAngle off ecliptic plane of the Earth's rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	x = angularMomentumMoon.x*momentumConverter;
	y = angularMomentumMoon.y*momentumConverter;
	z = angularMomentumMoon.z*momentumConverter;
	printf("\nAngular momentum of the Moon 				= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	printf("\nMagnitude of the angular momentum of the Moon 		= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	printf("\nAngle off ecliptic plane of the Moon's rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	x = centerOfMassEarthMoonSystem.x*lengthConverter;
	y = centerOfMassEarthMoonSystem.y*lengthConverter;
	z = centerOfMassEarthMoonSystem.z*lengthConverter;
	printf("\nCenter of mass of the Earth-Moon system 		= (%f, %f, %f)", x, y, z);
	
	x = centerOfMassEarth.x*lengthConverter;
	y = centerOfMassEarth.y*lengthConverter;
	z = centerOfMassEarth.z*lengthConverter;
	printf("\nCenter of mass of the Earth system 			= (%f, %f, %f)", x, y, z);
	
	x = centerOfMassMoon.x*lengthConverter;
	y = centerOfMassMoon.y*lengthConverter;
	z = centerOfMassMoon.z*lengthConverter;
	printf("\nCenter of mass of the Moon system 			= (%f, %f, %f)\n", x, y, z);
	
	x = linearVelocityEarthMoonSystem.x*velocityConverter;
	y = linearVelocityEarthMoonSystem.y*velocityConverter;
	z = linearVelocityEarthMoonSystem.z*velocityConverter;
	printf("\nLinear Velocity of the Earth-Moon system 		= (%f, %f, %f)", x, y, z);
	
	x = linearVelocityEarth.x*velocityConverter;
	y = linearVelocityEarth.y*velocityConverter;
	z = linearVelocityEarth.z*velocityConverter;
	printf("\nLinear Velocity of the Earth system 			= (%f, %f, %f)", x, y, z);
	
	x = linearVelocityMoon.x*velocityConverter;
	y = linearVelocityMoon.y*velocityConverter;
	z = linearVelocityMoon.z*velocityConverter;
	printf("\nLinear Velocity of the Moon system 			= (%f, %f, %f)\n", x, y, z);
	
	printf("\n*****Stats of the entire system to check the numerical scheme's validity*****\n");
	
	x = centerOfMassUniversalSystem.x*lengthConverter;
	y = centerOfMassUniversalSystem.y*lengthConverter;
	z = centerOfMassUniversalSystem.z*lengthConverter;
	printf("\nCenter of mass of the entire system 		        = (%f, %f, %f)\n", x, y, z);
	
	x = linearVelocityUniversalSystem.x*velocityConverter;
	y = linearVelocityUniversalSystem.y*velocityConverter;
	z = linearVelocityUniversalSystem.z*velocityConverter;
	printf("\nLinear velocity of the entire system system 		= (%f, %f, %f)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	printf("\nMagnitude of the linear velocity of the entire system 	= %f\n", mag);
	
	x = angularMomentumUniversalSystem.x*momentumConverter;
	y = angularMomentumUniversalSystem.y*momentumConverter;
	z = angularMomentumUniversalSystem.z*momentumConverter;
	printf("\nAngular momentum of the entire system system 		= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	printf("\nMagnitude of the angular momentum of the entire system 	= %e\n", mag);
	
	printf("\n*************************************************************************\n");
	
	printf("\n******************* Just the good stuff *********************************\n");
	
	printf("\n percent off correct Earth mass = %f ", 100.0*(massEarth*massConverter/(MassOfEarth)));
	printf("\n percent off correct Moon mass  = %f ", 100.0*(massMoon*massConverter/(MassOfMoon)));
	printf("\n\n Earth mass percent iron = %f mass percent silicate = %f", float(earthFeCountBody1*MassFe + earthFeCountBody2*MassFe)/massEarth, float(earthSiCountBody1*MassSi + earthSiCountBody2*MassSi)/massEarth);
	printf("\n Moon mass percent iron = %f mass percent silicate = %f", float(moonFeCountBody1*MassFe + moonFeCountBody2*MassFe)/massMoon, float(moonSiCountBody1*MassSi + moonSiCountBody2*MassSi)/massMoon);
	if((moonFeCountBody2 + moonSiCountBody2) != 0)
	{
		printf("\n\n Moon body1/body2 ratio     = %f ", float(moonFeCountBody1*MassFe + moonSiCountBody1*MassSi)/float(moonFeCountBody2*MassFe + moonSiCountBody2*MassSi));
	}
	
	x = angularMomentumEarthMoonSystem.x*momentumConverter;
	y = angularMomentumEarthMoonSystem.y*momentumConverter;
	z = angularMomentumEarthMoonSystem.z*momentumConverter;
	mag = sqrt(x*x + y*y + z*z);
	printf("\n Percent off correct angular momentum of the Earth-Moon System = %f ", 100.0*(1.0 - mag/AngularMomentumEarthMoonSystem));
	
	x = angularMomentumEarth.x*momentumConverter;
	y = angularMomentumEarth.y*momentumConverter;
	z = angularMomentumEarth.z*momentumConverter;
	mag = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/mag);
	printf("\n Percent off correct axial tilt of the Earth = %f ", 100.0*(1.0 - angle/EarthAxialTilt));
	
	
	printf("\n\n*************************************************************************\n\n\n");
}

void recordFinalCollisionStat(double time)
{
	double mag, size, angle, x, y, z;
	
	double timeConverter = UnitTime;
	double lengthConverter = UnitLength;
	double massConverter = UnitMass; 
	double velocityConverter = UnitLength/UnitTime; 
	double momentumConverter = UnitMass*UnitLength*UnitLength/UnitTime;
	
	findEarthAndMoon();
	int earthFeCountBody1 = 0;
	int earthFeCountBody2 = 0;
	int earthSiCountBody1 = 0;
	int earthSiCountBody2 = 0;
	int moonFeCountBody1 = 0;
	int moonFeCountBody2 = 0;
	int moonSiCountBody1 = 0;
	int moonSiCountBody2 = 0;
	
	float massUniversalSystem = getMassCollision(0);
	float massEarthMoonSystem = getMassCollision(1);
	float massEarth = getMassCollision(2);
	float massMoon = getMassCollision(3);
	
	float3 centerOfMassUniversalSystem = getCenterOfMassCollision(0);
	float3 centerOfMassEarthMoonSystem = getCenterOfMassCollision(1);
	float3 centerOfMassEarth = getCenterOfMassCollision(2);
	float3 centerOfMassMoon = getCenterOfMassCollision(3);
	
	float3 linearVelocityUniversalSystem = getLinearVelocityCollision(0);
	float3 linearVelocityEarthMoonSystem = getLinearVelocityCollision(1);
	float3 linearVelocityEarth = getLinearVelocityCollision(2);
	float3 linearVelocityMoon = getLinearVelocityCollision(3);
	
	float3 angularMomentumUniversalSystem = getAngularMomentumCollision(0);
	float3 angularMomentumEarthMoonSystem = getAngularMomentumCollision(1);
	float3 angularMomentumEarth = getAngularMomentumCollision(2);
	float3 angularMomentumMoon = getAngularMomentumCollision(3);
	
	for(int i = 0; i < NumberOfEarthElements; i++)
	{
		if(EarthIndex[i] < NFe1) 			earthFeCountBody1++;
		else if(EarthIndex[i] < NFe1 + NFe2) 		earthFeCountBody2++;
		else if(EarthIndex[i] < NFe1 + NFe2 + NSi1) 	earthSiCountBody1++;
		else 						earthSiCountBody2++;
	}
	
	for(int i = 0; i < NumberOfMoonElements; i++)
	{
		if(MoonIndex[i] < NFe1) 			moonFeCountBody1++;
		else if(MoonIndex[i] < NFe1 + NFe2) 		moonFeCountBody2++;
		else if(MoonIndex[i] < NFe1 + NFe2 + NSi1) 	moonSiCountBody1++;
		else 						moonSiCountBody2++;
	}
	
	fprintf(RunStatsFile,"\n\n\n*************************************************************************\n\n");
	fprintf(RunStatsFile,"\nThe following are the final stats of the run when time = %f hours\n", time*timeConverter/3600.0);
	fprintf(RunStatsFile,"\nDistance is measured in Kilometers");
	fprintf(RunStatsFile,"\nMass is measured in Kilograms");
	fprintf(RunStatsFile,"\nTime is measured in seconds");
	fprintf(RunStatsFile,"\nVelocity is measured in Kilometers/second");
	fprintf(RunStatsFile,"\nAngular momentun is measured in Kilograms*Kilometers*Kilometers/seconds\n");
	
	fprintf(RunStatsFile,"\nThe mass of Earth 		= %e", massEarth*massConverter);
	fprintf(RunStatsFile,"\nThe mass of Moon 		= %e", massMoon*massConverter);
	if(massMoon != 0.0) fprintf(RunStatsFile,"\nThe mass ratio Earth/Moon 	= %f\n", massEarth/massMoon);
	
	fprintf(RunStatsFile,"\nMoon iron from body 1 		= %d", moonFeCountBody1);
	fprintf(RunStatsFile,"\nMoon silicate from body 1 	= %d", moonSiCountBody1);
	fprintf(RunStatsFile,"\nMoon iron from body 2 		= %d", moonFeCountBody2);
	fprintf(RunStatsFile,"\nMoon silicate from body 2 	= %d", moonSiCountBody2);
	if((moonFeCountBody2 + moonSiCountBody2) == 0)
	{
		fprintf(RunStatsFile,"\nThe Moon is only composed of elements from body 1\n");
	}
	else if((moonFeCountBody1 + moonSiCountBody1) == 0)
	{
		fprintf(RunStatsFile,"\nThe Moon is only composed of elements from body 2\n");
	}
	else
	{
		fprintf(RunStatsFile,"\nMoon ratio body1/body2 		= %f\n", (float)(moonFeCountBody1 + moonSiCountBody1)/(float)(moonFeCountBody2 + moonSiCountBody2));
	}
	
	fprintf(RunStatsFile,"\nEarth iron from body 1 		= %d", earthFeCountBody1);
	fprintf(RunStatsFile,"\nEarth silicate from body 1 	= %d", earthSiCountBody1);
	fprintf(RunStatsFile,"\nEarth iron from body 2 		= %d", earthFeCountBody2);
	fprintf(RunStatsFile,"\nEarth silicate from body 2 	= %d", earthSiCountBody2);
	if((earthFeCountBody2 + earthSiCountBody2) == 0)
	{
		fprintf(RunStatsFile,"\nThe Earth is only composed of elements from body 1\n");
	}
	else if((earthFeCountBody1 + earthSiCountBody1) == 0)
	{
		fprintf(RunStatsFile,"\nThe Earth is only composed of elements from body 2\n");
	}
	else
	{

		fprintf(RunStatsFile,"\nEarth ratio body1/body2 		= %f\n", (float)(earthFeCountBody1 + earthSiCountBody1)/(float)(earthFeCountBody2 + earthSiCountBody2));
	}
	
	//It is always assumed that the ecliptic plane is the xz-plane.
	x = angularMomentumEarthMoonSystem.x*momentumConverter;
	y = angularMomentumEarthMoonSystem.y*momentumConverter;
	z = angularMomentumEarthMoonSystem.z*momentumConverter;
	fprintf(RunStatsFile,"\nAngular momentum of the Earth Moon system 		= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(RunStatsFile,"\nMagnitude of the angular momentum of the system 	= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	fprintf(RunStatsFile,"\nAngle off ecliptic plane of the system's rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	x = angularMomentumEarth.x*momentumConverter;
	y = angularMomentumEarth.y*momentumConverter;
	z = angularMomentumEarth.z*momentumConverter;
	fprintf(RunStatsFile,"\nAngular momentum of the Earth 				= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(RunStatsFile,"\nMagnitude of the angular momentum of the Earth 		= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	fprintf(RunStatsFile,"\nAngle off ecliptic plane of the Earth's rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	x = angularMomentumMoon.x*momentumConverter;
	y = angularMomentumMoon.y*momentumConverter;
	z = angularMomentumMoon.z*momentumConverter;
	fprintf(RunStatsFile,"\nAngular momentum of the Moon 				= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(RunStatsFile,"\nMagnitude of the angular momentum of the Moon 		= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	fprintf(RunStatsFile,"\nAngle off ecliptic plane of the Moon's rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	x = centerOfMassEarthMoonSystem.x*lengthConverter;
	y = centerOfMassEarthMoonSystem.y*lengthConverter;
	z = centerOfMassEarthMoonSystem.z*lengthConverter;
	fprintf(RunStatsFile,"\nCenter of mass of the Earth-Moon system 		= (%f, %f, %f)", x, y, z);
	
	x = centerOfMassEarth.x*lengthConverter;
	y = centerOfMassEarth.y*lengthConverter;
	z = centerOfMassEarth.z*lengthConverter;
	fprintf(RunStatsFile,"\nCenter of mass of the Earth system 			= (%f, %f, %f)", x, y, z);
	
	x = centerOfMassMoon.x*lengthConverter;
	y = centerOfMassMoon.y*lengthConverter;
	z = centerOfMassMoon.z*lengthConverter;
	fprintf(RunStatsFile,"\nCenter of mass of the Moon system 			= (%f, %f, %f)\n", x, y, z);
	
	x = linearVelocityEarthMoonSystem.x*velocityConverter;
	y = linearVelocityEarthMoonSystem.y*velocityConverter;
	z = linearVelocityEarthMoonSystem.z*velocityConverter;
	fprintf(RunStatsFile,"\nLinear Velocity of the Earth-Moon system 		= (%f, %f, %f)", x, y, z);
	
	x = linearVelocityEarth.x*velocityConverter;
	y = linearVelocityEarth.y*velocityConverter;
	z = linearVelocityEarth.z*velocityConverter;
	fprintf(RunStatsFile,"\nLinear Velocity of the Earth system 			= (%f, %f, %f)", x, y, z);
	
	x = linearVelocityMoon.x*velocityConverter;
	y = linearVelocityMoon.y*velocityConverter;
	z = linearVelocityMoon.z*velocityConverter;
	fprintf(RunStatsFile,"\nLinear Velocity of the Moon system 			= (%f, %f, %f)\n", x, y, z);
	
	fprintf(RunStatsFile,"\n*****Stats of the entire system to check the numerical scheme's validity*****\n");
	
	x = centerOfMassUniversalSystem.x*lengthConverter;
	y = centerOfMassUniversalSystem.y*lengthConverter;
	z = centerOfMassUniversalSystem.z*lengthConverter;
	fprintf(RunStatsFile,"\nCenter of mass of the entire system 		        = (%f, %f, %f)\n", x, y, z);
	
	x = linearVelocityUniversalSystem.x*velocityConverter;
	y = linearVelocityUniversalSystem.y*velocityConverter;
	z = linearVelocityUniversalSystem.z*velocityConverter;
	fprintf(RunStatsFile,"\nLinear velocity of the entire system system 		= (%f, %f, %f)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(RunStatsFile,"\nMagnitude of the linear velocity of the entire system 	= %f\n", mag);
	
	x = angularMomentumUniversalSystem.x*momentumConverter;
	y = angularMomentumUniversalSystem.y*momentumConverter;
	z = angularMomentumUniversalSystem.z*momentumConverter;
	fprintf(RunStatsFile,"\nAngular momentum of the entire system system 		= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(RunStatsFile,"\nMagnitude of the angular momentum of the entire system 	= %e\n", mag);
	
	fprintf(RunStatsFile,"\n*************************************************************************\n");
	
	fprintf(RunStatsFile,"\n******************* Just the good stuff *********************************\n");
	
	fprintf(RunStatsFile,"\n percent off correct Earth mass = %f ", 100.0*(massEarth*massConverter/(MassOfEarth)));
	fprintf(RunStatsFile,"\n percent off correct Moon mass  = %f ", 100.0*(massMoon*massConverter/(MassOfMoon)));
	fprintf(RunStatsFile,"\n\n Earth mass percent iron = %f mass percent silicate = %f", float(earthFeCountBody1*MassFe + earthFeCountBody2*MassFe)/massEarth, float(earthSiCountBody1*MassSi + earthSiCountBody2*MassSi)/massEarth);
	fprintf(RunStatsFile,"\n Moon mass percent iron = %f mass percent silicate = %f", float(moonFeCountBody1*MassFe + moonFeCountBody2*MassFe)/massMoon, float(moonSiCountBody1*MassSi + moonSiCountBody2*MassSi)/massMoon);
	if((moonFeCountBody2 + moonSiCountBody2) != 0)
	{
		fprintf(RunStatsFile,"\n\n Moon body1/body2 ratio     = %f ", float(moonFeCountBody1*MassFe + moonSiCountBody1*MassSi)/float(moonFeCountBody2*MassFe + moonSiCountBody2*MassSi));
	}
	
	x = angularMomentumEarthMoonSystem.x*momentumConverter;
	y = angularMomentumEarthMoonSystem.y*momentumConverter;
	z = angularMomentumEarthMoonSystem.z*momentumConverter;
	mag = sqrt(x*x + y*y + z*z);
	fprintf(RunStatsFile,"\n Percent off correct angular momentum of the Earth-Moon System = %f ", 100.0*(1.0 - mag/AngularMomentumEarthMoonSystem));
	
	x = angularMomentumEarth.x*momentumConverter;
	y = angularMomentumEarth.y*momentumConverter;
	z = angularMomentumEarth.z*momentumConverter;
	mag = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/mag);
	fprintf(RunStatsFile,"\n Percent off correct axial tilt of the Earth = %f ", 100.0*(1.0 - angle/EarthAxialTilt));
	
	fprintf(RunStatsFile,"\n\n*************************************************************************\n\n\n");
}

void recordPosAndVel()
{
	fwrite(Pos, sizeof(float4), N, PosAndVelFile);
	fwrite(Vel, sizeof(float4), N, PosAndVelFile);
}

void recordContinuePosAndVel(double time)
{
	fwrite(&time, sizeof(double), 1, ContinueRunPosAndVelFile);
	fwrite(Pos, sizeof(float4), N, ContinueRunPosAndVelFile);
	fwrite(Vel, sizeof(float4), N, ContinueRunPosAndVelFile);
}

void drawSimplePictureSeperate()
{
	float3 centerOfMass1 = getCenterOfMassSeperate(1);
	float3 centerOfMass2 = getCenterOfMassSeperate(2);
	float3 linearVelocity1 = getLinearVelocitySeperate(1);
	float3 linearVelocity2 = getLinearVelocitySeperate(2);
	float3 angularMomentum1 = getAngularMomentumSeperate(1, centerOfMass1, linearVelocity1);
	float3 angularMomentum2 = getAngularMomentumSeperate(2, centerOfMass2, linearVelocity2);
	float Stretch;
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	//Coloring all the elements 
	glBegin(GL_POINTS);
     		for(int i=0; i<N; i++)
		{
			if(i < NFe1) 
			{
		    		glColor3d(1.0,0.0,0.0);
			}
			else if(i < NFe1 + NSi1)
			{
				glColor3d(1.0,1.0,0.5);
			}
			else if(i < NFe1 + NSi1 + NFe2) 
			{
		    		glColor3d(1.0,0.0,1.0);
			}
			else
			{
				glColor3d(0.0,0.5,0.0);
			}
			
			glVertex3f(Pos[i].x, Pos[i].y, Pos[i].z);
		}
	glEnd();

	glLineWidth(1.0);
	//Placing a green vector in the direction of the disired linear motion of each body
	glColor3f(0.0,1.0,0.0);
	Stretch = 1.0;
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass1.x, centerOfMass1.y, centerOfMass1.z);
		glVertex3f(centerOfMass1.x + InitialVelocity1.x*Stretch, centerOfMass1.y + InitialVelocity1.y*Stretch, centerOfMass1.z + InitialVelocity1.z*Stretch);
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass2.x, centerOfMass2.y, centerOfMass2.z);
		glVertex3f(centerOfMass2.x + InitialVelocity2.x*Stretch, centerOfMass2.y + InitialVelocity2.y*Stretch, centerOfMass2.z + InitialVelocity2.z*Stretch);
	glEnd();
	
	//Placing a yellow vector in the direction of the actual linear motion of each body
	glColor3f(1.0,1.0,0.0);
	Stretch = 30.0;
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass1.x, centerOfMass1.y, centerOfMass1.z);
		glVertex3f(centerOfMass1.x + linearVelocity1.x*Stretch, centerOfMass1.y + linearVelocity1.y*Stretch, centerOfMass1.z + linearVelocity1.z*Stretch);
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass2.x, centerOfMass2.y, centerOfMass2.z);
		glVertex3f(centerOfMass2.x + linearVelocity2.x*Stretch, centerOfMass2.y + linearVelocity2.y*Stretch, centerOfMass2.z + linearVelocity2.z*Stretch);
	glEnd();
	
	//Placing a blue vector in the direction of the disired angular momentum 
	glColor3f(0.0,0.0,1.0);	
	Stretch = 50.0;
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass1.x, centerOfMass1.y, centerOfMass1.z);
		glVertex3f(centerOfMass1.x + InitialSpin1.x*Stretch, centerOfMass1.y + InitialSpin1.y*Stretch, centerOfMass1.z + InitialSpin1.z*Stretch);
	glEnd();
	
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass2.x, centerOfMass2.y, centerOfMass2.z);
		glVertex3f(centerOfMass2.x + InitialSpin2.x*Stretch, centerOfMass2.y + InitialSpin2.y*Stretch, centerOfMass2.z + InitialSpin2.z*Stretch);
	glEnd();
	
	//Placing a red vector in the direction of the actual angular momentum 
	glColor3f(1.0,0.0,0.0);	
	Stretch = 50.0;
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass1.x, centerOfMass1.y, centerOfMass1.z);
		glVertex3f(centerOfMass1.x + angularMomentum1.x*Stretch, centerOfMass1.y + angularMomentum1.y*Stretch, centerOfMass1.z + angularMomentum1.z*Stretch);
	glEnd();
	
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass2.x, centerOfMass2.y, centerOfMass2.z);
		glVertex3f(centerOfMass2.x + angularMomentum2.x*Stretch, centerOfMass2.y + angularMomentum2.y*Stretch, centerOfMass2.z + angularMomentum2.z*Stretch);
	glEnd();
	
	glutSwapBuffers();
}

void drawPictureCollision()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBegin(GL_POINTS);
     		for(int i=0; i<N; i++)
		{
			if(i < NFe1) 
			{
		    		glColor3d(1.0,0.0,0.0);
			}
			else if(i < NFe1 + NFe2)
			{
				glColor3d(1.0,0.0,1.0);
			}
			else if(i < NFe1 + NFe2 + NSi1) 
			{
				glColor3d(1.0,1.0,0.5);
			}
			else
			{
				glColor3d(0.0,0.5,0.0);
			}
			
			glVertex3f(Pos[i].x, Pos[i].y, Pos[i].z);
		}
	glEnd();
	
	glutSwapBuffers();
}

void drawAnalysisPictureCollision()
{
	int i;
	
	findEarthAndMoon();
	float massSystem = getMassCollision(0);
	float massEarth = getMassCollision(1);
	float massMoon = getMassCollision(2);
	float3 centerOfMassSystem = getCenterOfMassCollision(0);
	float3 centerOfMassEarth = getCenterOfMassCollision(1);
	float3 centerOfMassMoon = getCenterOfMassCollision(2);
	float3 linearVelocitySystem = getLinearVelocityCollision(0);
	float3 linearVelocityEarth = getLinearVelocityCollision(1);
	float3 linearVelocityMoon = getLinearVelocityCollision(2);
	float3 angularMomentumSystem = getAngularMomentumCollision(0);
	float3 angularMomentumEarth = getAngularMomentumCollision(1);
	float3 angularMomentumMoon = getAngularMomentumCollision(2);
	float Stretch;
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	//Coloring all the elements
	glPointSize(1.0);
	glBegin(GL_POINTS);
     		for(i=0; i<N; i++)
		{
			if(i < NFe1) 
			{
		    		glColor3d(1.0,0.0,0.0);
			}
			else if(i < NFe1 + NFe2)
			{
				glColor3d(1.0,0.0,1.0);
			}
			else if(i < NFe1 + NFe2 + NSi1) 
			{
				glColor3d(1.0,1.0,0.5);
			}
			else
			{
				glColor3d(0.0,0.5,0.0);
			}
			
			glVertex3f(Pos[i].x, Pos[i].y, Pos[i].z);
		}
	glEnd();
	
	glPointSize(1.0);
	//Recoloring the Earth elements blue
	glColor3d(0.0,0.0,1.0);
	glBegin(GL_POINTS);
		for(i = 0; i < NumberOfEarthElements; i++)
		{	
				glVertex3f(Pos[EarthIndex[i]].x, Pos[EarthIndex[i]].y, Pos[EarthIndex[i]].z);
		}
	glEnd();
	
	//Recoloring the Moon elements red
	glColor3d(1.0,0.0,0.0);
	glBegin(GL_POINTS);
		for(i = 0; i < NumberOfMoonElements; i++)
		{	
			glVertex3f(Pos[MoonIndex[i]].x, Pos[MoonIndex[i]].y, Pos[MoonIndex[i]].z);
		}
	glEnd();

	glLineWidth(1.0);
	//Placing green vectors in the direction of linear velocity of the Moon
	Stretch = 1.0;
	glColor3f(0.0,1.0,0.0);
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMassMoon.x, centerOfMassMoon.y, centerOfMassMoon.z);
		glVertex3f(	centerOfMassMoon.x + linearVelocityMoon.x*Stretch, 
				centerOfMassMoon.y + linearVelocityMoon.y*Stretch, 
				centerOfMassMoon.z + linearVelocityMoon.z*Stretch);
	glEnd();
	
	//Place a white point at the center of mass of the Earth-Moon system
	glColor3d(1.0,1.0,1.0);
	glPointSize(10.0);
	glBegin(GL_POINTS);
		glVertex3f(centerOfMassSystem.x, centerOfMassSystem.y, centerOfMassSystem.z);
	glEnd();
	
	//Place a yellow point at the center of mass of the Earth
	glColor3d(1.0,1.0,0.0);
	glPointSize(5.0);
	glBegin(GL_POINTS);
		glVertex3f(centerOfMassEarth.x, centerOfMassEarth.y, centerOfMassEarth.z);
	glEnd();
	
	//Place a yellow point at the center of mass of the Moon
	glColor3d(1.0,1.0,0.0);
	glPointSize(5.0);
	glBegin(GL_POINTS);
		glVertex3f(centerOfMassMoon.x, centerOfMassMoon.y, centerOfMassMoon.z);
	glEnd();
	
	//Placing white vectors in the direction of the angular momentum of the Earth-Moon system
	glColor3f(1.0,1.0,1.0);
	Stretch = 1.0;
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMassSystem.x, centerOfMassSystem.y, centerOfMassSystem.z);
		glVertex3f(	centerOfMassSystem.x + angularMomentumSystem.x*Stretch/massSystem, 
				centerOfMassSystem.y + angularMomentumSystem.y*Stretch/massSystem, 
				centerOfMassSystem.z + angularMomentumSystem.z*Stretch/massSystem);
	glEnd();
	
	//Placing blue vectors in the direction of the angular momentum of the Earth
	Stretch = 1.0;
	glBegin(GL_LINE_LOOP);
	glColor3f(0.0,0.0,1.0);
		glVertex3f(centerOfMassEarth.x, centerOfMassEarth.y, centerOfMassEarth.z);
		glVertex3f(	centerOfMassEarth.x + angularMomentumEarth.x*Stretch/massEarth, 
				centerOfMassEarth.y + angularMomentumEarth.y*Stretch/massEarth, 
				centerOfMassEarth.z + angularMomentumEarth.z*Stretch/massEarth);
	glEnd();
	
	//Placing red vectors in the direction of the angular momentum of the Moon
	Stretch = 1.0;
	glColor3f(1.0,0.0,0.0);
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMassMoon.x, centerOfMassMoon.y, centerOfMassMoon.z);
		glVertex3f(	centerOfMassMoon.x + angularMomentumMoon.x*Stretch/massMoon, 
				centerOfMassMoon.y + angularMomentumMoon.y*Stretch/massMoon, 
				centerOfMassMoon.z + angularMomentumMoon.z*Stretch/massMoon);
	glEnd();
	
	glutSwapBuffers();
	
	free(EarthIndex);
	free(MoonIndex);
}

void transformInitialConditionsFromSeperateToCollision()
{
	int k;
	cudaMemcpy( PlaceHolder, Pos_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Pos2");
	k = 0;
	for(int i = 0; i < NFe1; i++)
	{
		Pos[k] = PlaceHolder[i];
		k++;
	}
	for(int i = NFe1 + NSi1; i < NFe1 + NSi1 + NFe2; i++)
	{
		Pos[k] = PlaceHolder[i];
		k++;
	}
	for(int i = NFe1; i < NFe1 + NSi1; i++)
	{
		Pos[k] = PlaceHolder[i];
		k++;
	}
	for(int i = NFe1 + NSi1 + NFe2; i < N; i++)
	{
		Pos[k] = PlaceHolder[i];
		k++;
	}
	
	cudaMemcpy( PlaceHolder, Vel_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Vel");
	k = 0;
	for(int i = 0; i < NFe1; i++)
	{
		Vel[k] = PlaceHolder[i];
		k++;
	}
	for(int i = NFe1 + NSi1; i < NFe1 + NSi1 + NFe2; i++)
	{
		Vel[k] = PlaceHolder[i];
		k++;
	}
	for(int i = NFe1; i < NFe1 + NSi1; i++)
	{
		Vel[k] = PlaceHolder[i];
		k++;
	}
	for(int i = NFe1 + NSi1 + NFe2; i < N; i++)
	{
		Vel[k] = PlaceHolder[i];
		k++;
	}
}

void nBodySeperate()
{ 
	float time = 0.0;
	int   tdraw = 1;
	
	int dampCheck = 0;
	int rest1Check = 0;
	int spinCheck = 0;
	
    	cudaMemcpy( Pos_DEV0, Pos, N *sizeof(float4), cudaMemcpyHostToDevice );
    	errorCheck("cudaMemcpy Pos3");
    	cudaMemcpy( Vel_DEV0, Vel, N *sizeof(float4), cudaMemcpyHostToDevice );
    	errorCheck("cudaMemcpy Vel");
   
	while(time < SetupTime)
	{	
		getForcesSeperate<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, ForceSeperateConstant);
		if(time < DampTime) 
		{
			if(dampCheck == 0)
			{
				printf("\n************************************************** Damping is on\n");
				dampCheck = 1;
				tdraw = 0;
			}
			moveBodiesDampedSeperate<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, MoveSeperateConstant, DampRateBody1, DampRateBody2);
		}
		else if(time < DampTime + DampRestTime)
		{
			if(rest1Check == 0)
			{
				printf("\n************************************************** Damp rest stage is on\n");
				rest1Check = 1;
				tdraw = 0;
			}
			moveBodiesSeperate<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, MoveSeperateConstant);
		}
		else
		{
			if(spinCheck == 0)
			{
				cudaMemcpy( Pos, Pos_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
				errorCheck("cudaMemcpy Pos4");
				cudaMemcpy( Vel, Vel_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
				errorCheck("cudaMemcpy Vel");
				spinBodySeperate(1, InitialSpin1);
				spinBodySeperate(2, InitialSpin2);
				cudaMemcpy( Pos_DEV0, Pos, N *sizeof(float4), cudaMemcpyHostToDevice );
				errorCheck("cudaMemcpy Pos5");
	    			cudaMemcpy( Vel_DEV0, Vel, N *sizeof(float4), cudaMemcpyHostToDevice );
	    			errorCheck("cudaMemcpy Vel");
				printf("\n************************************************** bodies have been spun\n");
				printf("\n************************************************** spin rest stage is on\n");
				spinCheck = 1;
			}
			moveBodiesSeperate<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, MoveSeperateConstant);
		}
    
		if(tdraw == DrawRate) 
		{
		    	cudaMemcpy( Pos, Pos_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
		    	errorCheck("cudaMemcpy Pos6");
		    	cudaMemcpy( Vel, Vel_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
		    	errorCheck("cudaMemcpy Vel");
		    	drawSimplePictureSeperate();
			//drawPictureSeperate();
			printf("\nSetup time in hours = %f\n", time*UnitTime/3600.0);
			tdraw = 0;
		}
		tdraw++;
		
		time += Dt;
	}
}

void resetInitialConditions()
{
	cudaMemcpy( Pos, Pos_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Pos7");
	cudaMemcpy( Vel, Vel_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Vel");
	setBodyPositionSeperate(1, InitialPosition1.x, InitialPosition1.y, InitialPosition1.z);
	setBodyVelocitySeperate(1, InitialVelocity1.x, InitialVelocity1.y, InitialVelocity1.z);
	setBodyPositionSeperate(2, InitialPosition2.x, InitialPosition2.y, InitialPosition2.z);
	setBodyVelocitySeperate(2, InitialVelocity2.x, InitialVelocity2.y, InitialVelocity2.z);
	printf("\n************************************************** Initial velocities have been given\n");
	cudaMemcpy( Pos_DEV0, Pos, N *sizeof(float4), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy Pos8");
    	cudaMemcpy( Vel_DEV0, Vel, N *sizeof(float4), cudaMemcpyHostToDevice );
    	errorCheck("cudaMemcpy Vel");
	printf("\n************************************************** The bodies have been created and intialized\n");
}	

void copyCreatedBodiesUpToDevice()
{	
	if(NumberOfGpus == 1 || UseMultipleGPU == 0)
	{
		cudaMemcpy( Pos_DEV0, Pos, N *sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy Pos9");
		cudaMemcpy( Vel_DEV0, Vel, N *sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy Vel");
	}
	else
	{
		cudaSetDevice(0);
		errorCheck("cudaSetDevice 0");
		cudaMemcpyAsync( PosFstHalf_0, Pos, (N/2)*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpyAsync PosFstHalf 0");
		cudaMemcpyAsync( PosSndHalf_0, Pos+(N/2), (N/2)*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpyAsync PosSndHalf 0");
		cudaMemcpyAsync( VelFstHalf_0, Vel, (N/2)*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpyAsync VelFstHalf 0");
		cudaMemcpyAsync( VelSndHalf_0, Vel+(N/2), (N/2)*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpyAsync VelSndHalf 0");
		
		cudaSetDevice(1);
		errorCheck("cudaSetDevice 0");
		cudaMemcpyAsync( PosFstHalf_1, Pos, (N/2)*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpyAsync PosFstHalf 0");
		cudaMemcpyAsync( PosSndHalf_1, Pos+(N/2), (N/2)*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpyAsync PosSndHalf 0");
		cudaMemcpyAsync( VelFstHalf_1, Vel, (N/2)*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpyAsync VelFstHalf 0");
		cudaMemcpyAsync( VelSndHalf_1, Vel+(N/2), (N/2)*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpyAsync VelSndHalf 0");
	}
}

double nBodyCollisionSingleGPU()
{ 
	int   tDraw = 1; 
	int   tRecord = 1;
		
	while(RunTime <= TotalRunTime)
	{
		getForcesCollisionSingleGPU<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, ForceCollisionConstant);
		moveBodiesCollisionSingleGPU<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, MoveCollisionConstant);
		
		if(tDraw == DrawRate) 
		{
			cudaMemcpy( Pos, Pos_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			cudaMemcpy( Vel, Vel_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");
			if	(DrawQuality == 1) drawAnalysisPictureCollision(); 
			else if	(DrawQuality == 2) drawPictureCollision();
			else 
			{
				printf("\nTSU Error: Invalid draw quality\n");
				exit(0);
			}
			tDraw = 0;
			printf("\nCollision run time = %f hours\n", RunTime*UnitTime/3600.0);
		}
		tDraw++;
		
		if(PrintCollisionStats == 1) 
		{
			cudaMemcpy( Pos, Pos_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			cudaMemcpy( Vel, Vel_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");	
			printCollisionStatsToScreen(RunTime);
			PrintCollisionStats = 0;
		}
		
		if(PrintContinueStats == 1) 
		{
			cudaMemcpy( Pos, Pos_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			cudaMemcpy( Vel, Vel_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");
			printContinueStatsToScreen(RunTime);	
			PrintContinueStats = 0;
		}
				
		if(WriteToFile == 1 && tRecord == RecordRate) 
		{
			cudaMemcpy( Pos, Pos_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			cudaMemcpy( Vel, Vel_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");
			recordPosAndVel();	
			tRecord = 0;
		}
		tRecord++;
		
		RunTime += Dt;
	}
	RunTime = RunTime - Dt;
	
	cudaMemcpy( Pos, Pos_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpyAsync Pos");
	cudaMemcpy( Vel, Vel_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpyAsync Vel");
	
	return(RunTime);
}

double nBodyCollisionDoubleGPU()
{ 
	int   tDraw = 1; 
	int   tRecord = 1;
	cout << "\nCollision run time start = " << RunTime*UnitTime/3600.0 << " hours." << endl;
	
	while(RunTime <= TotalRunTime)
	{
		cudaSetDevice(0);
		errorCheck("cudaSetDevice 0");
		getForcesCollisionDoubleGPU0<<<GridConfig, BlockConfig>>>(PosFstHalf_0, PosSndHalf_0, VelFstHalf_0, VelSndHalf_0,   ForceFstHalf_0, N, ForceCollisionConstant);
		errorCheck("getForcesCollisionDoubleGPU 0");
		moveBodiesCollisionDoubleGPU0<<<GridConfig, BlockConfig>>>(PosFstHalf_0,  VelFstHalf_0,   ForceFstHalf_0,  N, MoveCollisionConstant);
		errorCheck("moveBodiesCollisionDoubleGPU 0");
		
		cudaSetDevice(1);
		errorCheck("cudaSetDevice 1");
		getForcesCollisionDoubleGPU1<<<GridConfig, BlockConfig>>>(PosFstHalf_1, PosSndHalf_1, VelFstHalf_1, VelSndHalf_1,   ForceSndHalf_1,  N, ForceCollisionConstant);
		errorCheck("getForcesCollisionDoubleGPU 1");
		moveBodiesCollisionDoubleGPU1<<<GridConfig, BlockConfig>>>(PosSndHalf_1,  VelSndHalf_1,   ForceSndHalf_1,  N, MoveCollisionConstant);
		errorCheck("moveBodiesCollisionDoubleGPU 1");
		
		cudaDeviceSynchronize();
		errorCheck("cudaDeviceSynchronize 1");

		cudaSetDevice(0);
		errorCheck("cudaSetDevice 0");
		cudaMemcpyPeerAsync(PosFstHalf_1,1,PosFstHalf_0,0,(N/2)*sizeof(float4));
		errorCheck("cudaMemcpyPeerAsync 0 - Pos");
		cudaMemcpyPeerAsync(VelFstHalf_1,1,VelFstHalf_0,0,(N/2)*sizeof(float4));
		errorCheck("cudaMemcpyPeerAsync 0 - Vel");
		
		cudaDeviceSynchronize();
		errorCheck("cudaDeviceSynchronize 2");
		
		cudaSetDevice(1);
		errorCheck("cudaSetDevice 1");
		cudaMemcpyPeerAsync(PosSndHalf_0,0,PosSndHalf_1,1,(N/2)*sizeof(float4));
		errorCheck("cudaMemcpyPeerAsync 1 - Pos");
		cudaMemcpyPeerAsync(VelSndHalf_0,0,VelSndHalf_1,1,(N/2)*sizeof(float4));
		errorCheck("cudaMemcpyPeerAsync 1 - Vel");
		
		cudaDeviceSynchronize();
		errorCheck("cudaDeviceSynchronize 3");
		
		if(tDraw == DrawRate) 
		{
			cudaSetDevice(0);
			errorCheck("cudaSetDevice 0");
			cudaMemcpyAsync(Pos, PosFstHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			cudaMemcpyAsync(Pos+(N/2), PosSndHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			
			cudaSetDevice(1);
			errorCheck("cudaSetDevice 1");
			cudaMemcpyAsync(Vel, VelFstHalf_1, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");
			cudaMemcpyAsync(Vel+(N/2), VelSndHalf_1,  (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");
			
			if	(DrawQuality == 1) drawAnalysisPictureCollision(); 
			else if	(DrawQuality == 2) drawPictureCollision();
			else 
			{
				printf("\nTSU Error: Invalid draw quality\n");
				exit(0);
			}
			tDraw = 0;
			cout << "\nCollision run time = " << RunTime*UnitTime/3600.0 << " hours." << endl;
		}
		tDraw++;
		
		if(PrintCollisionStats == 1) 
		{
			cudaSetDevice(0);
			errorCheck("cudaSetDevice 0");
			cudaMemcpyAsync(Pos, PosFstHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			cudaMemcpyAsync(Pos+(N/2), PosSndHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			
			cudaSetDevice(1);
			errorCheck("cudaSetDevice 1");
			cudaMemcpyAsync(Vel, VelFstHalf_1, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");
			cudaMemcpyAsync(Vel+(N/2), VelSndHalf_1,  (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");
			
			printCollisionStatsToScreen(RunTime);
			PrintCollisionStats = 0;
		}
		
		if(PrintContinueStats == 1) 
		{
			cudaSetDevice(0);
			errorCheck("cudaSetDevice 0");
			cudaMemcpyAsync(Pos, PosFstHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			cudaMemcpyAsync(Pos+(N/2), PosSndHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			
			cudaSetDevice(1);
			errorCheck("cudaSetDevice 1");
			cudaMemcpyAsync(Vel, VelFstHalf_1, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");
			cudaMemcpyAsync(Vel+(N/2), VelSndHalf_1,  (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");
			
			printContinueStatsToScreen(RunTime);
			PrintContinueStats = 0;
		}
		
		if(WriteToFile == 1 && tRecord == RecordRate) 
		{
			cudaSetDevice(0);
			errorCheck("cudaSetDevice 0");
			cudaMemcpyAsync(Pos, PosFstHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			cudaMemcpyAsync(Pos+(N/2), PosSndHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Pos");
			
			cudaSetDevice(1);
			errorCheck("cudaSetDevice 1");
			cudaMemcpyAsync(Vel, VelFstHalf_1, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");
			cudaMemcpyAsync(Vel+(N/2), VelSndHalf_1,  (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
			errorCheck("cudaMemcpyAsync Vel");
		
			recordPosAndVel();	
			tRecord = 0;
		}
		tRecord++;
	
		RunTime += Dt;
	}
	RunTime = RunTime -Dt;
	cout << "\nCollision run time end = " << RunTime*UnitTime/3600.0 << " hours." << endl;
	
	cudaSetDevice(0);
	errorCheck("cudaSetDevice 0");
	cudaMemcpyAsync(Pos, PosFstHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpyAsync Pos");
	cudaMemcpyAsync(Pos+(N/2), PosSndHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpyAsync Pos");
	
	cudaSetDevice(1);
	errorCheck("cudaSetDevice 1");
	cudaMemcpyAsync(Vel, VelFstHalf_1, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpyAsync Vel");
	cudaMemcpyAsync(Vel+(N/2), VelSndHalf_1,  (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpyAsync Vel");
	
	return(RunTime);
}

void cleanKill(double time)
{
	if(NumberOfGpus == 1 || UseMultipleGPU == 0) 
	{
		cudaMemcpy( Pos, Pos_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
		errorCheck("cudaMemcpyAsync Pos");
		cudaMemcpy( Vel, Vel_DEV0, N *sizeof(float4), cudaMemcpyDeviceToHost );
		errorCheck("cudaMemcpyAsync Vel");
	}
	else
	{
		cudaSetDevice(0);
		errorCheck("cudaSetDevice 0");
		cudaMemcpyAsync(Pos, PosFstHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
		errorCheck("cudaMemcpyAsync Pos");
		cudaMemcpyAsync(Pos+(N/2), PosSndHalf_0, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
		errorCheck("cudaMemcpyAsync Pos");
	
		cudaSetDevice(1);
		errorCheck("cudaSetDevice 1");
		cudaMemcpyAsync(Vel, VelFstHalf_1, (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
		errorCheck("cudaMemcpyAsync Vel");
		cudaMemcpyAsync(Vel+(N/2), VelSndHalf_1,  (N/2)*sizeof(float4), cudaMemcpyDeviceToHost );
		errorCheck("cudaMemcpyAsync Vel");
	}
	
	recordFinalCollisionStat(time);
	
	recordContinuePosAndVel(time);
	
	printContinueStatsToFile(time);
	
	cleanUpCollision();
	exit(0);
}

static void signalHandler(int signum)
{
	int command;
    
	cout << "\n\n******************************************************" << endl;
	cout << "Enter:666 to kill the run." << endl;
	cout << "Enter:1 to cleanly terminate the run.\t(not valid in the setup stage)." << endl;
	cout << "Enter:2 to change the draw rate." << endl;
	cout << "Enter:3 to change the draw quality.\t(not valid in the setup stage)." << endl;
	cout << "Enter:4 to set your eye location." << endl;
	cout << "Enter:5 to set the Center of Mass as your center." << endl;
	cout << "Enter:6 to print the run stats.\t(not valid in the setup stage)." << endl;
	cout << "Enter:7 to print the continue stats.\t(not valid in the setup stage)." << endl;
	cout << "Enter:8 to change the total run time." << endl;
	cout << "Enter:9 to continue the run." << endl;
	cout << "******************************************************\n\nCommand: ";
    
	cin >> command;
    
	if(command == 666)
	{
	
		cout << "\n\n******************************************************" << endl;
		cout << "Are you sure you want to terminate the run?" << endl;
		cout << "Enter:666 again if you are sure. Enter anything else to continue the run." << endl;
		cout << "******************************************************\n\nCommand: ";
		cin >> command;
		
		if(command == 666)
		{
			cleanUpCollision();
			exit(0);
		}
	}
	else if(command == 1)
	{
		cleanKill(RunTime);
	}
	else if(command == 2)
	{
		cout << "\nEnter the desired draw rate: ";
		cin >> DrawRate;
		cout << "\nDrawRate: " << DrawRate << endl;
	}
	else if(command == 3)
	{
		cout << "\nEnter the desired draw quality.\n1 for analysis.\n2 for standard." << endl;
		cin >> DrawQuality;
		cout << "\nDrawQuality: " << DrawQuality << endl;
	}
	else if (command == 4)
	{
    	cout << "******************************************************" << endl;
		cout << "Here is where your current Eye is at: " << endl;
		cout << "EyeX: " << EyeX << endl;
		cout << "EyeY: " << EyeY << endl;
		cout << "EyeZ: " << EyeZ << endl;
		cout << "Changing this will determine how close/far you are." << endl;
    	cout << "******************************************************" << endl;
		cout << "\nEnter the desired x location of your eye (double): ";
		cin >> EyeX;
		cout << "Enter the desired y location of your eye (double): ";
		cin >> EyeY;
		cout << "Enter the desired z location of your eye (double): ";
		cin >> EyeZ;
	    	
	    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glLoadIdentity();
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
		glMatrixMode(GL_MODELVIEW);
		gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
	    	//glutPostRedisplay();
	    	//Display();
	}
	else if (command == 5)
	{
		float3 temp = getCenterOfMassCollision(0);
		cout << "******************************************************" << endl;
		cout << "Center of Mass in the X-direction: " << temp.x << endl;
		cout << "Center of Mass in the Y-direction: " << temp.y << endl;
		cout << "Center of Mass in the Z-direction: " << temp.z << endl;
		cout << "This is the Center of Mass of the System" << endl;
    	cout << "******************************************************" << endl;
		
		CenterX = temp.x;
		CenterY = temp.y;
		CenterZ = temp.z;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glLoadIdentity();
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
		glMatrixMode(GL_MODELVIEW);
		gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
	    	//glutPostRedisplay();
	    	//Display();
	}
	else if (command == 6)
	{
		PrintCollisionStats = 1;
	}
	else if (command == 7)
	{
		PrintContinueStats = 1;
	}
	else if (command == 8)
	{
		cout << "\nEnter the desired TotalRunTime (float): ";
		cin >> TotalRunTime;

		TotalRunTime *= 3600.0/UnitTime;
	}
	else if (command == 9)
	{
		cout << "\nRun continued." << endl;
	}
	else
	{
		cout <<"\n\n Invalid Command\n" << endl;
	}
}

void typeOfRunCheck() 
{
	cout << "\nEnter 0 to create a new Run.\nEnter 1 to create a branch Run.\nEnter 2 to continue an existing Run.\n\n";
	cin >> TypeOfRun;
}

void readRootStartPosAndVelFile()
{
	FILE *temp = fopen("RootStartPosAndVel","rb");
	fread(Pos, sizeof(float4), N, temp);
	fread(Vel, sizeof(float4), N, temp);
	fclose(temp);
	
	fseek(PosAndVelFile,0,SEEK_END);
}

void readContinuePosAndVel()
{
	ContinueRunPosAndVelFile = fopen("ContinueRunPosAndVel","rb");
	fread(&RunTime, sizeof(double), 1, ContinueRunPosAndVelFile);
	fread(Pos, sizeof(float4), N, ContinueRunPosAndVelFile);
	fread(Vel, sizeof(float4), N, ContinueRunPosAndVelFile);
	//ContinueRunPosAndVelFile.clear();
	fclose(ContinueRunPosAndVelFile);
}

void control()
{	
	double time;
	struct sigaction sa;
	
	sa.sa_handler = signalHandler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART; // Restart functions if interrupted by handler
	if (sigaction(SIGINT, &sa, NULL) == -1)
	{
		printf("\nTSU Error: sigaction error\n");
	}

	//Setup run
	if (TypeOfRun == 0) {
		createFolderForNewRun();
		readRunParameters();
		setRunParameters();
		openNewRunFiles();
		recordSetupStats();
		loadKernalConstantStructures();
		allocateCPUMemory();
		checkSetupForErrors();
	
		//Create and initialize bodies
		deviceSetupSeperate();	
		createBodies();	
		nBodySeperate();
		resetInitialConditions();    	
		recordStatsOfCreatedBodies(); 
		recordStartPosVelOfCreatedBodiesSeperate();  	
		transformInitialConditionsFromSeperateToCollision();    	
		cleanUpSeperate();
	
		//Collide bodies
		deviceSetupCollision();
		copyCreatedBodiesUpToDevice();
	
		if(NumberOfGpus == 1 || UseMultipleGPU == 0) time = nBodyCollisionSingleGPU();
		else time = nBodyCollisionDoubleGPU();
	
		recordFinalCollisionStat(time);
		recordContinuePosAndVel(time);
		printContinueStatsToFile(time);
		cleanUpCollision();
		printf("\n DONE \n");
		exit(0);
	}
	else if (TypeOfRun == 1) 
	{
		createFolderForBranchRun(RootFolderName);
		readRunParameters();
		setRunParameters();
		readBranchParameters();
		setBranchParameters();
		
		openBranchRunFiles();
		
		allocateCPUMemory();
		
		readRootStartPosAndVelFile();
		
		InitialPosition1.x += BranchPosition1.x;
		InitialPosition1.y += BranchPosition1.y;
		InitialPosition1.z += BranchPosition1.z;
		
		InitialPosition2.x += BranchPosition2.x;
		InitialPosition2.y += BranchPosition2.y;
		InitialPosition2.z += BranchPosition2.z;
		
		InitialVelocity1.x += BranchVelocity1.x;
		InitialVelocity1.y += BranchVelocity1.y;
		InitialVelocity1.z += BranchVelocity1.z;
		
		InitialVelocity2.x += BranchVelocity2.x;
		InitialVelocity2.y += BranchVelocity2.y;
		InitialVelocity2.z += BranchVelocity2.z;
		
		InitialSpin1.x += BranchSpin1.x;
		InitialSpin1.y += BranchSpin1.y;
		InitialSpin1.z += BranchSpin1.z;
		InitialSpin1.w += BranchSpin1.w;
		
		InitialSpin2.x += BranchSpin2.x;
		InitialSpin2.y += BranchSpin2.y;
		InitialSpin2.z += BranchSpin2.z;
		InitialSpin2.w += BranchSpin2.w;
		
		recordSetupStats();
		
		loadKernalConstantStructures();
		checkSetupForErrors();
		deviceSetupSeperate();
		
		//From here down to nBodySeperate is like the create bodies above but all that needs to be done is move and spin 
		setBodyPositionSeperate(1, InitialPosition1.x, InitialPosition1.y, InitialPosition1.z);
		//setBodyVelocitySeperate(1, InitialVelocity1.x, InitialVelocity1.y, InitialVelocity1.z);
		setBodyPositionSeperate(2, InitialPosition2.x, InitialPosition2.y, InitialPosition2.z);
		//setBodyVelocitySeperate(2, InitialVelocity2.x, InitialVelocity2.y, InitialVelocity2.z);
		
		//This is really the added spin but must be put in initail to fool nBodySeperate because the original spin is already done
		InitialSpin1 = BranchSpin1;
		InitialSpin2 = BranchSpin2;
		
		DampTime = -1.0;
		DampRestTime = -1.0;
		SetupTime = BranchSpinRestTime;
		
		nBodySeperate();
		
		resetInitialConditions();    	
		recordStatsOfCreatedBodies(); 
		recordStartPosVelOfCreatedBodiesSeperate();  	
		transformInitialConditionsFromSeperateToCollision();    	
		cleanUpSeperate();
	
		//Collide bodies
		TotalRunTime = BranchRunTime;
		deviceSetupCollision();
		copyCreatedBodiesUpToDevice();
	
		if(NumberOfGpus == 1 || UseMultipleGPU == 0) time = nBodyCollisionSingleGPU();
		else time = nBodyCollisionDoubleGPU();
	
		recordFinalCollisionStat(time);
		recordContinuePosAndVel(time);
		printContinueStatsToFile(time);
		cleanUpCollision();
		printf("\n DONE \n");
		exit(0);
	}
	else if (TypeOfRun == 2)
	{
		chdir(RootFolderName);
		
		readRunParameters();
		setRunParameters();
		
		loadKernalConstantStructures();
		allocateCPUMemory();
		checkSetupForErrors();
		
		readContinuePosAndVel();
		
		openContinueRunFiles();

		TotalRunTime = AddedRunTime*3600.0/UnitTime + RunTime;
		
		//Collide bodies
		deviceSetupCollision();
		copyCreatedBodiesUpToDevice();
		
		if(NumberOfGpus == 1 || UseMultipleGPU == 0) time = nBodyCollisionSingleGPU();
		else time = nBodyCollisionDoubleGPU();
	
		recordFinalCollisionStat(time);
		recordContinuePosAndVel(time);
		printContinueStatsToFile(time);
		cleanUpCollision();
		printf("\n DONE \n");
		exit(0);
	}
	else
	{
		printf("\n Bad TypeOfRun value \n");
		exit(0);
	}
}

//https://www.opengl.org/archives/resources/faq/technical/viewing.htm
void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(Left, Right, Bottom, Top, Front, Back);
	glMatrixMode(GL_MODELVIEW);
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
}

void reshape(GLint w, GLint h) 
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(Left, Right, Bottom, Top, Front, Back);
	glMatrixMode(GL_MODELVIEW);
}

void init()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(Left, Right, Bottom, Top, Front, Back);
	glMatrixMode(GL_MODELVIEW);
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

int main(int argc, char** argv)
{
	if( argc < 1)
	{
		printf("\n You need to intire the run type (int 0 new run, 1 branch run, or 2 continue run) on the comand line\n"); 
		exit(0);
	}
	else
	{
		TypeOfRun = atoi(argv[1]);
	}
	
	if( TypeOfRun == 1)
	{
		if(argc < 2)
		{
			printf("\n You need to intire a root folder to work from on the comand line\n");
			exit(0);
		}
		else
		{
			strcat(RootFolderName, argv[2]);
		}
	}
	
	if( TypeOfRun == 2)
	{
		if(argc < 2)
		{
			printf("\n You need to intire a root folder to work from on the comand line\n");
			exit(0);
		}
		else 
		{
			strcat(RootFolderName, argv[2]);
		}
		
		if(argc < 3)
		{
			printf("\n You need to intire the extra run time for the continuation\n");
			exit(0);
		}
		else 
		{
			AddedRunTime = atof(argv[3]);
		}
	}
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Giant Impact Hypothesis Simulation");
	
	glutReshapeFunc(reshape);
	
	init();
	
	glShadeModel(GL_SMOOTH);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutIdleFunc(control);
	glutMainLoop();
	return 0;
}






