#include <stdio.h>
#include <cmath>
#include <hydrasub/hydrabase/lists.H>
#include <hydrasub/hydrabase/linalg.H>
#include <hydrasub/hydrabase/mcmc.H>


enum VARTYPE {VAR_RESPONSE=0, VAR_LIN=1, VAR_FIXED=2, VAR_RANDOM=3};
char *vartypes[]={"VAR_RESPONSE","VAR_LIN","VAR_FIXED","VAR_RANDOM"};
VARTYPE *variable_type=NULL;
int *num_cat=NULL;

enum PARTYPE {PAR_BETA,PAR_LSIGMA,PAR_EPSILON, PAR_OVERDISPERSION};
char *partypes[]={"PAR_BETA","PAR_LSIGMA","PAR_EPSILON", "PAR_OVERDISPERSION"};

PARTYPE *param_type=NULL;
int *epsilon_lsigma=NULL;
int *param_data_index=NULL;
int *data_param_index=NULL;
int *param_origdata_index=NULL;
VARTYPE *data_vartype=NULL;
bool *kid_related=NULL;

bool egg_betabinom=true, kid_betabinom=true;

char *initfile=NULL, **parname=NULL;

char **prior_col_names=NULL, prior_file_name[1000]="";
int prior_num_rows=0, prior_num_cols=0;
double **specific_priors=NULL;

void init(int numparams,double *params, 
	  int num_hyper,double *hyper_parameters)
{
  int initlen=0;
  char **initparname=NULL;
  double *init_mu=NULL, *init_sd=NULL;

  double lsigma_mu=hyper_parameters[0];
  double lsigma_sd=hyper_parameters[1];

  if(initfile)
    {
      FILE *f=fopen(initfile,"r");
      if(f)
	{
	  char *str=new char[1000];
	  int numrow=0;
	  size_t nn=999;
	  getline(&str,&nn,f);
	  while(!feof(f))
	    {	      
	      while(!feof(f))
		{
		  if(str[0]!='\0' && str[0]!='#')
		    numrow++;

		  nn=999;
		  getline(&str,&nn,f);
		}
	    }
	  fclose(f);

	  initlen=numrow;
	  initparname=new char*[initlen];
	  init_mu=new double[initlen];
	  init_sd=new double[initlen];
	  
	  f=fopen(initfile,"r");
	  nn=999;
	  getline(&str,&nn,f);
	  int i=0;
	  while(!feof(f))
	    {	      
	      while(!feof(f))
		{
		  if(str[0]!='\0' && str[0]!='#')
		    {
		      initparname[i]=new char[200];
		      char currpname[200];
		      double mu,sd;
		      sscanf(str,"%s %lf %lf", &currpname, &mu, &sd); 
		      //printf("Init read: %s %f %f\n", 
		      //currpname, mu, sd);
		      if(mu<(-100) | mu>100)
			mu=0.0;
		      if(sd>100)
			sd=1.0;
		      strcpy(initparname[i],currpname);
		      init_mu[i]=mu;
		      init_sd[i]=sd;
		      //printf("Init set : %s %f %f\n", 
		      //initparname[i], init_mu[i], init_sd[i]);
		      i++;
		    }
		  
		  nn=999;
		  getline(&str,&nn,f);
		}
	    }
	  fclose(f);
	  
	  delete [] str;
	}
    }

  for(int i=0;i<numparams;i++)
    {
      bool found_init=false;
      if(initlen>0)
	{
	  for(int j=0;j<initlen;j++)
	    if(!strcmp(parname[i],initparname[j]) && 
	       strncmp(parname[i],"epsilon",7))
	      {
		params[i]=init_mu[j]+init_sd[j]*gauss();
		//printf("Parameter %s initialized to %f "
		//     "(mu=%f sd=%f) from init file\n",
		//     parname[i], params[i], init_mu[j], init_sd[j]);
		found_init=true;
	      }
	}
      
      if(!strncmp(parname[i],"epsilon",7))
	params[i]=0.01*gauss();
      else if(!strncmp(parname[i],"lsigma",6))
	params[i]=lsigma_mu+0.5*lsigma_sd*gauss();
      else if(!found_init)
	params[i]=gauss();
    }

  if(initparname)
    doubledelete(initparname,initlen);
  if(init_mu)
    delete [] init_mu;
  if(init_sd)
    delete [] init_sd;
}

double lognorm(double x, double m, double s)
{
  return(-0.5*log(2.0*M_PI)-log(s)-0.5*(x-m)*(x-m)/s/s);
}

double logprior(int numparams, double *params,
		int num_hyperparameters,double *hyper_parameters)
{
  int i;
  double lp=0.0;
  
  double lsigma_mu=hyper_parameters[0];
  double lsigma_sd=hyper_parameters[1];
  double loverdisp_mu=hyper_parameters[2];
  double loverdisp_sd=hyper_parameters[3];
  double beta_mu=hyper_parameters[4];
  double beta_sd=hyper_parameters[5];
  
  for(i=0;i<numparams;i++)
    {
      switch(param_type[i])
	{
	case PAR_BETA:
	  if(specific_priors==NULL || !strncmp(parname[i],"beta0",5))
	    lp += lognorm(params[i], beta_mu, beta_sd);
	  else
	    {
	      int j;
	      bool prior_found=false;
	      for(j=0;j<prior_num_rows && ! prior_found;j++)
		{
		  //printf("%d %d\n", param_origdata_index[i],int(specific_priors[j][0]));
		  if(param_origdata_index[i]==int(specific_priors[j][0]))
		    {
		      prior_found=true;
		      break;
		    }
		}
	      if(!prior_found)
		{
		  printf("Failed to find prior for parameter number %d (%s), "
			 "effect number %d in the priors file %s!", 
			 i, parname[i], param_origdata_index[i], prior_file_name);
		  exit(0);
		}
	      beta_mu=0.0;
	      beta_sd=log(specific_priors[j][2])/1.96/specific_priors[j][1];
	      //printf("%s_sd = %f, mult=%f range=%f\n", parname[i], beta_sd,
	      //     specific_priors[j][2],specific_priors[j][1]); 

	      lp += lognorm(params[i], beta_mu, beta_sd);
	    }
	  break;
	case PAR_LSIGMA:
	  lp += lognorm(params[i], lsigma_mu, lsigma_sd);
	  break;
	case PAR_EPSILON:
	  lp += lognorm(params[i], 0.0, 
			exp(params[epsilon_lsigma[i]]));
	  break;
	case PAR_OVERDISPERSION:
	  lp += lognorm(params[i], loverdisp_mu, loverdisp_sd); 
	  break;
	default:
	  printf("Unknown parameter type:%d\n",
		 int(param_type[i]));
	  exit(1);
	  break;
	}
    }

  return(lp);
}

double invlogit(double x)
{
  if(x< -100.0)
    return 0.0;
  if(x> +100.0)
    return 1.0;
  
  return 1.0/(1.0+exp(-x));
}

double logbinom(int k, int n, double p)
{
  if(p==1.0)
    {
      if(k==n)
	return 0.0;
      else
	return -1e+200;
    }
  if(p==0.0)
    {
      if(k==0)
	return 0.0;
      else
	return -1e+200;
    }

  double ret=lgamma(double(n+1))-lgamma(double(k+1))-lgamma(double(n-k+1))+
    double(k)*log(p)+double(n-k)*log(1.0-p);

  if(!(ret>-1e+200 && ret<1e+200))
    return -1e+200;
  else
    return ret;
}

bool check_lgamma(double x)
{
  if(lgamma(x)<x && x>10) // If this is the case, something is wrong!
    return false;
  else
    return true;
}

double logbetabinom(int k, int n, double p, double overdisp)
{
  if(p==1.0)
    {
      if(k==n)
	return 0.0;
      else
	return -1e+200;
    }
  if(p==0.0)
    {
      if(k==0)
	return 0.0;
      else
	return -1e+200;
    }

  double alpha=p*overdisp;
  double beta=(1.0-p)*overdisp;
  
  if(!check_lgamma(double(n+1)) || !check_lgamma(double(k+1)) || 
     !check_lgamma(double(n-k+1)) || !check_lgamma(double(n)+alpha+beta) ||
     !check_lgamma(double(k)+alpha) || !check_lgamma(double(n-k)+beta) ||
     !check_lgamma(alpha+beta) || !check_lgamma(alpha) || !check_lgamma(beta))
    return -1e+200;
  
  double ret=
    lgamma(double(n+1))-lgamma(double(k+1))-lgamma(double(n-k+1))-
    lgamma(double(n)+alpha+beta)+lgamma(double(k)+alpha)+lgamma(double(n-k)+beta)+
    lgamma(alpha+beta)-lgamma(alpha)-lgamma(beta);
  
  if(!(ret>-1e+200 && ret<1e+200))
    return -1e+200;
  else
    return ret;
}


bool firstwrong=true;
double loglik(double **data, int numrows, int numcols,
	      int numparams, double *params) 
{
  int i,j,k=0;
  double ll=0.0, ll_prev=0.0;
  double egg_overdisp, kid_overdisp;
  
  if(egg_betabinom)
    {
      egg_overdisp=exp(params[k]);
      k++;
    }

  if(kid_betabinom)
    {
      kid_overdisp=exp(params[k]);
      k++;
    }
  
  for(i=0;i<numrows;i++)
    {
      double pred_egg=params[k];
      double pred_kids=params[k+1];
      
      if(!(pred_egg>-1e+200 && pred_egg<1e+200))
	{
	  printf("Noe er galt1, pred_egg=%f params[0]=%f\n",
		 pred_egg, params[k]);
	}

      // Check column indexes for data, kid_related etc:
      for(j=4;j<numcols;j++)
	{
	  switch(data_vartype[j])
	    {
	    case VAR_LIN:
	      if(!kid_related[data_param_index[j]])
		pred_egg  += params[data_param_index[j]] * data[i][j];
	      else
		pred_kids += params[data_param_index[j]] * data[i][j];
	      break;
	    case VAR_FIXED:
	      //printf("i=%d %f\n", i, data[i][j]);
	      if(data[i][j]!=0.0)
		{
		  if(!kid_related[data_param_index[j]])
		    pred_egg  += params[data_param_index[j]+int(data[i][j])-1];
		  else
		    pred_kids += params[data_param_index[j]+int(data[i][j])-1];
		  //printf("%s=%f\n", parname[data_param_index[j]+int(data[i][j])-1],
		  //params[data_param_index[j]+int(data[i][j])-1]);
		}
	      break;
	    case VAR_RANDOM:
	      if(!kid_related[data_param_index[j]])
		pred_egg  += params[data_param_index[j]+int(data[i][j])+1];
	      else
		pred_kids += params[data_param_index[j]+int(data[i][j])+1];
	      break;
	    default:
	      printf("Unknown variable type:%d\n",
		     int(data_vartype[j]));
	      exit(1);
	      break;
	    }
	}
      
      int egg, egg_start=data[i][0], egg_end=data[i][1], 
	egg_interval=egg_end-egg_start+1;
      if(egg_interval <= 0)
	{
	  printf("egg_interval=%d for i=%d!\n",egg_interval,i);
	  exit(0);
	}
      int kids,kids_start=data[i][2], kids_end=data[i][3], 
	kids_interval=kids_end-kids_start+1;   
      if(kids_interval <= 0)
	{
	  printf("kids_interval=%d for i=%d!\n",kids_interval,i);
	  exit(0);
	}

      double prob_outcome=0.0;
      for(egg=egg_start;egg<=egg_end;egg++)
	{
	  double prob_egg=(egg_betabinom && egg_overdisp<10000.0) ? 
	    exp(logbetabinom(egg, 7, invlogit(pred_egg),egg_overdisp)) : 
	    exp(logbinom(egg, 7, invlogit(pred_egg)));
	  
	  int kids_end2=kids_end;
	  if(egg<kids_end2)
	    kids_end2=egg;
	  
	  double prob_kids=0.0;
	  for(kids=kids_start;kids<=kids_end2;kids++)
	    prob_kids += (kid_betabinom & kid_overdisp<10000.0) ? 
	      exp(logbetabinom(kids, egg, invlogit(pred_kids),kid_overdisp)) : 
	      exp(logbinom(kids, egg, invlogit(pred_kids)));
	  
	  prob_outcome += prob_egg*prob_kids;
	}
      ll_prev=ll;
      ll+=log(prob_outcome);
	    
      if(!(ll>-1e+200 && ll<1e+200) && (ll_prev>-1e+200 && ll_prev<1e+200))
	{
	  /* printf("Noe er galt5! ll=%f ll_prev=%f L_egg_sum=%f l_egg_max=%g "
		 "p_egg=%f pred_egg=%f egg_start=%d egg_end=%d "
		 "data[i=%d][0]=%f data[i=%d][1]=%f data[i=%d][4]=%f egg_interval=%d\n",
		 ll,ll_prev,L_egg_sum,l_egg_max,p_egg,pred_egg,egg_start,egg_end,
		 i,data[i][0],i,data[i][1],i,data[i][4],egg_interval);

	  L_egg_sum=0.0;
	  for(egg=egg_start;egg<=egg_end;egg++)
	    L_egg_sum+=exp(l_egg[egg-egg_start]-l_egg_max)/double(egg_interval);
	  */

	  return(-1e+200);
	}
    }
  
  if(ll>20000.0 && firstwrong)
    {
      firstwrong=false;
      printf("%g\n",loglik(data, numrows, numcols,numparams, params));
    }

  return(ll);
}


double **read_csv(char *filename, char ***column_names, 
		  int *numrows, int *numcols, bool silent)
{
  FILE *f=fopen(filename,"r");
  if(!f)
    {
      printf("Failed to open file \"%s\"!", filename);
      exit(2);
    }
  
  char *str=new char[100000];
  size_t nn=99999;
  
  getline(&str,&nn,f);
  
  int i,j,k,numcol=1,numrow=0,slen=strlen(str);
  if(!silent)
    printf("%d\n",slen);

  for(i=0;i<slen;i++)
    if(str[i]==';')
      numcol++;
  
  nn=99999;
  getline(&str,&nn,f);
  while(!feof(f))
    {
      if(str[0]!='\0')
	numrow++;

      nn=99999;
      getline(&str,&nn,f);
    }
  fclose(f);

  if(!silent)
    printf("numrow=%d numcol=%d\n",numrow,numcol);

  f=fopen(filename,"r");
  if(!f)
    {
      printf("Failed to open file \"%s\"!", filename);
      exit(2);
    }

  getline(&str,&nn,f);
 
  char **cname=new char*[numcol];
  i=j=k=0;
  while(i<slen)
    {
      cname[k]=new char[100];
      j=0;
      while(i<slen && str[i]!=';')
	{
	  if(str[i]!='\"' && str[i]!='\n')
	    cname[k][j++]=str[i];
	  i++;
	}
      cname[k][j]='\0';
      k++;
      i++;
    }
  
  if(!silent)
    printf("numcol=%d k=%d\n",numcol,k);
  
  double **vals=make_matrix(numrow,numcol);
  char *currval=new char[10000];
  
  nn=99999;
  getline(&str,&nn,f);
  int len=0;
  while(!feof(f))
    {
      i=j=k=0;
      slen=strlen(str);
      while(i<slen)
	{
	  j=0;
	  while(i<slen && str[i]!=';')
	    {
	      if(str[i]!='\"')
		currval[j++]=str[i];
	      i++;
	    }
	  currval[j]='\0';
	  if(k==numcol)
	    {
	      printf("len=%d numrow=%d numcol=%d k=%d\n",len,numrow,numcol,k);
	      exit(0);
	    }
	  if(strlen(currval)>0)
	    vals[len][k]=atof(currval);
	  else
	    vals[len][k]=MISSING_VALUE;
	  //printf("%d %d %s %f\n",len,k,currval, vals[len][k]);
	  
	  k++;
	  i++;
	}
      
      len++;
      nn=99999;
      getline(&str,&nn,f);
    }
  fclose(f);  
  
  *column_names=cname;
  *numcols=numcol;
  *numrows=numrow;
  
  delete [] str;
  delete [] currval;
  
  return vals;
} 

void usage(void)
{
  printf("Usage: mcmc_egg_kids_betabin [options] <file> "
	 "[list of variable numbers to include]\n");
  printf("The list should contain the number of the variables to use, ranging\n");
  printf("from 1 to number of variables in the csv file.\n");
  printf("Use positive numbers for egg variables and negative numbers for kids\n");
  printf("variables.\n");
  printf("Keep in mind that factorial variables needs tbe started with \n");
  printf("ffactor for fixed factors and rfactor for random factors.\n");
  printf("Egg and kids intervals should be names \"egg.start\", \"egg.end\",\n");
  printf("\"kids.start\" and \"kids.end\".\n");
  printf("Uses per default the beta-binomial distribution, reparametrizes as\n");
  printf("to contain the probability (explained by the regression) and the\n");
  printf("overdispersion parameter (alpha+beta). When the overdispersion parameter\n");
  printf("is much larger than the upper limit for the outcome (7 for eggs, "
	 "number of eggs for kids),\n");
  printf("that entails little overdispersion, while if it is low, there's much "
	 "overdispersion."); 
  printf("\n");
  printf("Options:\n");
  printf("        -t : Toggles talkative MCMC mode.\n");
  printf("        -T <number of tempering chains>\n");
  printf("        -g : toggles graphs. Should only be used for low number of \n");
  printf("             parameters. Random effects ignored.\n");
  printf("        -P : Toggles off showing parameter summary\n");
  printf("        -N <number of samples> : Number of MCMC samples (spacing=10).\n");
  printf("             Burnin=10*number of samples. Default, N=400\n");
  printf("        -h : Show usage and exit\n");
  printf("        -e : Switches off egg overdispersion\n");
  printf("        -k : Switches off kid overdispersion\n");
  printf("        -i <initial value file>: A file giving a proposal for initial.\n");
  printf("           values for each parameter.\n");
  printf("           Format: <parameter name> <mcmc_mean> <mcmc_sd>\n");
  printf("        -p <prior file>: Determine priors by a file containing covariate\n");
  printf("             number, range and maximal (95%%) multiplicative effect on odds.\n");
  exit(0);
}

int main(int argc, char **argv)
{
  if(argc<2)
    usage();
  
  randify();

  int numtemp=1, N=400;
  bool show_plots=false, show_param_summary=true, silent=true;
  int i,j;

  while(argv[1][0]=='-')
    {
      switch(argv[1][1])
	{
	case 'p':
	  {
	    strcpy(prior_file_name, argv[2]);
	    specific_priors=read_csv(prior_file_name, &prior_col_names, 
				     &prior_num_rows, &prior_num_cols, true);
	    /*
	    for(i=0;i<prior_num_rows;i++)
	      {
		for(j=0;j<prior_num_cols;j++)
		  if(j==0)
		    printf("%d ", int(specific_priors[i][j]));
		  else
		    printf("%8.4f ", specific_priors[i][j]);
		printf("\n");
	      }
	    */
	    
	    argv++;
	    argc--;
	    break;
	  }
	case 'i':
	  initfile=argv[2];
	  argc--;
	  argv++;
	  break;
	case 'k':
	  kid_betabinom=false;
	  break;
	case 'e':
	  egg_betabinom=false;
	  break;
	case 'N':
	  N=atoi(argv[2]);
	  argc--;
	  argv++;
	  break;
	case 'P':
	  show_param_summary=false;
	  break;
	case 't':
	  silent=false;
	  break;
	case 'g':
	  show_plots=true;
	  break;
	case 'T':
	  numtemp=atoi(argv[2]);
	  argc--;
	  argv++;
	  break;
	case 'h':
	  usage();
	  exit(0);
	default:
	  printf("Unknown option!");
	  exit(4);
	}
      argc--;
      argv++;
    }

  char **cname;
  int numrow, numcol;
  double **vals=read_csv(argv[1], &cname, &numrow, &numcol, silent);
  
  if(!vals || numrow<=0 || numcol<=0)
    {
      printf("Could not fetch contents from file "
	     "\"%s\"!\n",argv[1]);
      exit(1);
    }
  
  num_cat=new int[numcol];
  variable_type=new VARTYPE[numcol];
  for(i=0;i<numcol;i++)
    {
      if(!strcmp(cname[i],"egg.start") | !strcmp(cname[i],"egg.end") |
	 !strcmp(cname[i],"kids.start") | !strcmp(cname[i],"kids.end"))
	{
	  variable_type[i]=VAR_RESPONSE;
	  num_cat[i]=0;
	}
      else if(!strncasecmp(cname[i],"ffactor_",8))
	{
	  variable_type[i]=VAR_FIXED;
	  char strbuffer[1000];
	  strcpy(strbuffer,cname[i]);
	  strcpy(cname[i],strbuffer+8);
	  double *cats=new double[numrow];
	  for(j=0;j<numrow;j++)
	    cats[j]=round(vals[j][i]);
	  num_cat[i]=1+(int) round(find_statistics(cats,numrow,MAX));
	  delete [] cats;
	}
      else if(!strncasecmp(cname[i],"rfactor_",8))
	{
	  variable_type[i]=VAR_RANDOM;
	  char strbuffer[1000];
	  strcpy(strbuffer,cname[i]);
	  strcpy(cname[i],strbuffer+8);
	  double *cats=new double[numrow];
	  for(j=0;j<numrow;j++)
	      cats[j]=round(vals[j][i]);
	  num_cat[i]=1+(int) round(find_statistics(cats,numrow,MAX));
	  delete [] cats;
	}
      else
	{
	  variable_type[i]=VAR_LIN;
	  num_cat[i]=0;
	}
    }
  
  argc-=2;
  argv+=2;
  int numparam=2; // intercepts

  // Count overdispersion parameters:
  if(egg_betabinom)
    numparam++;
  if(kid_betabinom)
    numparam++;
  
  for(i=0;i<argc;i++)
    {
      int var=atoi(argv[i]);
      int absvar=var>0 ? var : -var;

      if(var==0)
	{
	  printf("Error: variable number 0 is not valid!\n");
	  exit(3);
	}
      
      if(absvar>numcol)
	{
	  printf("Error: variable number (%d) exceeds the number of "
		 "variables in the csv file (%d)!\n",var,numcol);
	  exit(3);
	}
      
      absvar--;
      
      if(variable_type[absvar]==VAR_LIN)
	numparam++;
      else if(variable_type[absvar]==VAR_FIXED)
	numparam+=num_cat[absvar]-1; // category 0 is control
      else if(variable_type[absvar]==VAR_RANDOM)
	numparam+=num_cat[absvar]+1; // sigma + one for each category
      else
	{
	  printf("Unknown variable type! vartype=%d", (int)variable_type[absvar]);
	  exit(10);
	}
      
      if(specific_priors && (variable_type[absvar]==VAR_LIN || 
			     variable_type[absvar]==VAR_FIXED))
	{
	  bool var_found=false;
	  for(j=0;j<prior_num_rows && ! var_found;j++)
	    if(int(specific_priors[j][0])==(absvar+1))
	      var_found=true;

	  if(!var_found)
	    {
	      printf("Variable number %d specified as covariate was "
		     "not found in the prior file, %s!\n", absvar+1, prior_file_name);
	      exit(0);
	    }
	}
      
      // two for each category
    }
  
  // Parameter names:
  parname=new char*[numparam];
  
  // Dataset used in the analysis:
  double **data=make_matrix(numrow, 4+argc);
  
  // Find response variable place in the input dataset:
  int egg_start_index=0, egg_end_index=0;
  int kids_start_index=0, kids_end_index=0;
  for(j=0;j<numcol;j++)
    {
      if(!strcmp(cname[j],"egg.start"))
	egg_start_index=j;
      if(!strcmp(cname[j],"egg.end"))
	egg_end_index=j;
      if(!strcmp(cname[j],"kids.start"))
	kids_start_index=j;
      if(!strcmp(cname[j],"kids.end"))
	kids_end_index=j;
    }

  // Fill out the response variables in the first 4 columns of the
  // dataset used for analysis:
  for(i=0;i<numrow;i++)
    {
      data[i][0]=vals[i][egg_start_index];
      data[i][1]=vals[i][egg_end_index];
      data[i][2]=vals[i][kids_start_index];
      data[i][3]=vals[i][kids_end_index];
    }
  
  // Should contain how to interpret each column of the 
  // dataset used for analysis:
  data_param_index=new int[4+argc];
  data_vartype=new VARTYPE[4+argc];
  
  // Should contain how to interpret each parameter:
  param_data_index=new int[numparam];
  param_origdata_index=new int[numparam];
  epsilon_lsigma=new int[numparam];
  param_type=new PARTYPE[numparam];
  kid_related=new bool[numparam];
  
  // Fill out first four columns of the interpretation
  // of the dataset used for analysis:
  data_param_index[0]=data_param_index[1]=0;
  data_param_index[2]=data_param_index[3]=1;
  for(i=0;i<4;i++)
    data_vartype[i]=VAR_RESPONSE;
  
  int k=0;
  
  // Overdispersion parameters:
  if(egg_betabinom)
    {
      parname[k]=new char[100];
      sprintf(parname[k],"logoverdispersion_%s","egg");
      param_type[k]=PAR_OVERDISPERSION;
      epsilon_lsigma[k]=0;
      param_data_index[k]=0;
      param_origdata_index[k]=0;
      kid_related[k]=0;
      k++;
    }
  
  if(kid_betabinom)
    {
      parname[k]=new char[100];
      sprintf(parname[k],"logoverdispersion_%s","kid");
      param_type[k]=PAR_OVERDISPERSION;
      epsilon_lsigma[k]=0;
      param_data_index[k]=2;
      param_origdata_index[k]=0;
      kid_related[k]=1;
      k++;
    }
  
  // First after overdispersion are two parameters that are the two 
  // intercepts, for eggs and kids:
  parname[k]=new char[100];
  sprintf(parname[k],"beta0_%s","egg");
  param_type[k]=PAR_BETA;
  epsilon_lsigma[k]=0;
  param_data_index[k]=0;
  param_origdata_index[k]=0;
  kid_related[k]=0;
  k++;
  
  parname[k]=new char[100];
  sprintf(parname[k],"beta0_%s","kid");
  param_type[k]=PAR_BETA;
  epsilon_lsigma[k]=0;
  param_data_index[k]=2;
  param_origdata_index[k]=0;
  kid_related[k]=1;
  k++;
  
  // Run through the user-specified effects:
  for(i=0;i<argc;i++)
    {
      int var,absvar;
      
      var=atoi(argv[i]);
      absvar=var>0 ? var : -var;
      absvar--;
      
      for(j=0;j<numrow;j++)
	data[j][4+i]=vals[j][absvar];
      data_param_index[4+i]=k;
      data_vartype[4+i]=variable_type[absvar];
      
      if(variable_type[absvar]==VAR_LIN)
	{
	  parname[k]=new char[100];
	  kid_related[k]=var<0;
	  sprintf(parname[k],"beta_%s_%s",kid_related[k] ? "kid" : "egg",
		  cname[absvar]);
	  param_type[k]=PAR_BETA;
	  param_data_index[k]=4+i;
	  param_origdata_index[k]=absvar+1;
	  epsilon_lsigma[k]=0;
	  k++;
	}
      else if(variable_type[absvar]==VAR_FIXED)
	{
	  for(j=1;j<num_cat[absvar];j++)
	    {
	      parname[k]=new char[100];
	      kid_related[k]=var<0;
	      sprintf(parname[k],"beta_%s_%s_%d",kid_related[k] ? "kid" : "egg",
		  cname[absvar],j);
	      param_type[k]=PAR_BETA;
	      param_data_index[k]=4+i;
	      param_origdata_index[k]=absvar+1;
	      epsilon_lsigma[k]=0;
	      k++;
	    }
	}
      else if(variable_type[absvar]==VAR_RANDOM)
	{
	  int lsigma_index=k;
	  parname[k]=new char[100];
	  kid_related[k]=var<0;
	  sprintf(parname[k],"lsigma_%s_%s",kid_related[k] ? "kid" : "egg",
		  cname[absvar]);
	  param_type[k]=PAR_LSIGMA;
	  param_data_index[k]=4+i;
	  param_origdata_index[k]=absvar+1;
	  epsilon_lsigma[k]=0;
	  k++;
	  
	  for(j=0;j<num_cat[absvar];j++)
	    {
	      parname[k]=new char[100];
	      kid_related[k]=var<0;
	      sprintf(parname[k],"epsilon_%s_%s_%d",
		      kid_related[k] ? "kid" : "egg", cname[absvar],j);
	      param_type[k]=PAR_EPSILON;
	      param_data_index[k]=4+i;
	      param_origdata_index[k]=absvar+1;
	      epsilon_lsigma[k]=lsigma_index;
	      k++;
	    }
	}
    }
  
  if(!silent)
    {
      /*
      for(i=0;i<numrow;i++)
	{
	  printf("%04d:",i+1);
	  for(j=0;j<4+argc;j++)
	    printf(" %f",data[i][j]);
	  printf("\n");
	}
      */
      
      printf("4+argc=%d\n",4+argc);
      for(i=0;i<4+argc;i++)
	printf("%d: %d %s\n", i, data_param_index[i],
	       vartypes[int(data_vartype[i])]);
      
      printf("numparam=%d\n",numparam);
      for(i=0;i<numparam;i++)
	printf("%s %s %d %d kidsrelated=%d\n",
	       parname[i],partypes[int(param_type[i])],epsilon_lsigma[i], 
	       param_data_index[i],kid_related[i]);
    }
  
  double *T=new double[numtemp];
  for(i=0;i<numtemp;i++)
    if(i==0)
      T[i]=1.0;
    else
      T[i]=pow(1.03, double(i));
  
  double *hyper=new double[6];
  double lsigma_mu=hyper[0]=(log(log(1.1))+log(log(3.0)))/2.0;
  double lsigma_sd=hyper[1]=(log(log(3.0))-log(log(1.1)))/2.0/1.96;
  double loverdisp_mu=hyper[2]=log(100);
  double loverdisp_sd=hyper[3]=log(100.0);
  double beta_mu=hyper[4]=0.0;
  double beta_sd=hyper[5]=100.0;
  
  /*  DEBUG
  double *par=new double[numparam];
  par[0]=484.818;
  par[1]=2.1706;
  par[2]=1.12148;
  par[3]=-1.3687;
  par[4]=1.06458;
  for(i=5;i<numparam;i++)
    par[i]=0.0;
  
  double lp=logprior(numparam, par,
		     6,hyper);
  double ll=loglik(data, numrow, 4+argc, numparam, par);
  printf("lp=%g ll=%g lp+ll=%g\n",lp, ll, lp+ll);
  exit(0);
  */

  double **res=mcmc(N, N*10, 10, numtemp, data, numrow, 4+argc, numparam, parname, 
		    T, 0.1, 6, hyper, init, logprior, loglik, silent,false,
		    false);
  
  if(show_param_summary)
    {
      for(i=0;i<numparam;i++)
	if(strncasecmp(parname[i],"epsilon",7))
	  {
	    double *par=new double[N];
	    for(j=0;j<N;j++)
	      par[j]=res[j][i];
	    show_mcmc_parameter(par,N, parname[i], !show_plots);
	    delete [] par;
	  }
    }
  
  double modellik=0;
  if(numparam<20)
    modellik=log_model_likelihood_multinormal(5*N*numtemp*numparam, res, numparam, N, 
					      data, numrow, 4+argc, 4, hyper, 
					      logprior, loglik, silent, false);
  else
    modellik=log_model_likelihood_indepnormal(5*N*numtemp*numparam, res, numparam, N, 
					      data, numrow, 4+argc, 4, hyper, 
					      logprior, loglik, silent);
  
  if(!silent)
    printf("Model log-likelihood=%7.3f\n", modellik);
  else
    printf("%7.3f\n", modellik);

  doubledelete(vals,numrow);
  doubledelete(data,numrow);
  doubledelete(parname,numparam);
  doubledelete(cname, numcol);
  delete [] T;
  delete [] hyper;
  doubledelete(res,N);
}
