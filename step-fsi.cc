/**
  Chiara Hergl
 Leibniz Universität Hannover (LUH)
 Institut für Angewandte Mathematik (IfAM)
 AG Wissenschaftliches Rechnen (GWR)

 Date: Nov 14, 2026
 E-mail: hergl@ifam.uni-hannover.de


 This code is a modification of 
 the ANS article open-source version:

 http://media.archnumsoft.org/10305/

 This code is based on the deal.II.9.5.1.

 deal.II step: fluid-structure interaction
 Keywords: fluid-structure interaction, nonlinear harmonic MMPDE, 
           finite elements, benchmark computation, 
     monolithic framework 

*/

// Include files
//--------------

// The first step, as always, is to include
// the functionality of these 
// deal.II library files and some C++ heade
// files.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>  

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
//#include <deal.II/grid/tria_boundary_lib.h> // old deal.II version 
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
//#include <dofs/dof_constraints.h>
//#include <deal.II/lac/constraint_matrix.h> // old deal.II versioncell_qp
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

//#include <numerics/vectors.h>
//#include <numerics/matrices.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>

//Debug
#include <csignal>   // für signal()
#include <iostream>  // für std::cerr etc.
#include <cstdlib>   // für std::abort()

// C++
#include <fstream>
#include <sstream>

// At the end of this top-matter, we import
// all deal.II names into the global
// namespace:       
using namespace dealii;

// First, we define tensors for the solution variables
// v (velocity), u (displacement), p (pressure),
// and w (second displacment). Moreover, we define 
// corresponding tensors for derivatives (e.g., gradients, 
// deformation gradients) and
// linearized tensors that are needed to solve the 
// non-linear problem with Newton's method.   
namespace ALE_Transformations
{    
  template <int dim> 
  inline
  double get_co (unsigned int q, std::vector<Vector<double> > old_solution_values)
  {      
    return old_solution_values[q](dim+dim+1);      
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_pI (unsigned int q, std::vector<Vector<double> > old_solution_values)
  {
    Tensor<2,dim> tmp;
    tmp[0][0] =  old_solution_values[q](dim+dim);
    tmp[1][1] =  old_solution_values[q](dim+dim);
      
    return tmp;      
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_pI_LinP (const double phi_i_p)
  {
    Tensor<2,dim> tmp;
    tmp.clear();
    tmp[0][0] = phi_i_p;    
    tmp[1][1] = phi_i_p;
      
    return tmp;
  }

  template <int dim> 
  inline
  Tensor<1,dim> get_grad_p (unsigned int q, std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)   
  {     
    Tensor<1,dim> grad_p;     
    grad_p[0] =  old_solution_grads[q][dim+dim][0];
    grad_p[1] =  old_solution_grads[q][dim+dim][1];
    
    return grad_p;
  }

  template <int dim> 
  inline
  Tensor<1,dim> get_grad_p_LinP (const Tensor<1,dim> phi_i_grad_p)   
  {
    Tensor<1,dim> grad_p;      
    grad_p[0] =  phi_i_grad_p[0];
    grad_p[1] =  phi_i_grad_p[1];
     
    return grad_p;
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_grad_u (unsigned int q, std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)   
  {   
    Tensor<2,dim> structure_continuation;     
    structure_continuation[0][0] = old_solution_grads[q][dim][0];
    structure_continuation[0][1] = old_solution_grads[q][dim][1];
    structure_continuation[1][0] = old_solution_grads[q][dim+1][0];
    structure_continuation[1][1] = old_solution_grads[q][dim+1][1];

    return structure_continuation;
  }

  template <int dim> 
  inline
  Tensor<2,dim> 
  get_grad_v (unsigned int q, std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)  
  {      
    Tensor<2,dim> grad_v;      
    grad_v[0][0] =  old_solution_grads[q][0][0];
    grad_v[0][1] =  old_solution_grads[q][0][1];
    grad_v[1][0] =  old_solution_grads[q][1][0];
    grad_v[1][1] =  old_solution_grads[q][1][1];
      
    return grad_v;
  }

  template <int dim> 
  inline
  Tensor<1,dim> get_grad_co (unsigned int q, std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)  
  {      
    Tensor<1,dim> grad_co;      
    grad_co[0] =  old_solution_grads[q][dim+dim+1][0];
    grad_co[1] =  old_solution_grads[q][dim+dim+1][1];
      
    return grad_co;
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_grad_v_T (const Tensor<2,dim> tensor_grad_v)
  {   
    Tensor<2,dim> grad_v_T;
    grad_v_T = transpose (tensor_grad_v);
            
    return grad_v_T;      
  }
  
  template <int dim> 
  inline
  Tensor<2,dim> get_grad_v_LinV (const Tensor<2,dim> phi_i_grads_v)  
  {     
    Tensor<2,dim> tmp;     
    tmp[0][0] = phi_i_grads_v[0][0];
    tmp[0][1] = phi_i_grads_v[0][1];
    tmp[1][0] = phi_i_grads_v[1][0];
    tmp[1][1] = phi_i_grads_v[1][1];
      
    return tmp;
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_Identity ()
  {   
    Tensor<2,dim> identity;
    identity[0][0] = 1.0;
    identity[0][1] = 0.0;
    identity[1][0] = 0.0;
    identity[1][1] = 1.0;
            
    return identity;      
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_F (unsigned int q, std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
  {     
    Tensor<2,dim> F;
    F[0][0] = 1.0 +  old_solution_grads[q][dim][0];
    F[0][1] = old_solution_grads[q][dim][1];
    F[1][0] = old_solution_grads[q][dim+1][0];
    F[1][1] = 1.0 + old_solution_grads[q][dim+1][1];
    return F;
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_F_T (const Tensor<2,dim> F)
  {
    return  transpose (F);
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_F_Inverse (const Tensor<2,dim> F)
  {     
    return invert (F);    
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_F_Inverse_T (const Tensor<2,dim> F_Inverse)
  { 
    return transpose (F_Inverse);
  }

  template <int dim> 
  inline
  double get_J (const Tensor<2,dim> tensor_F)
  {     
    return determinant (tensor_F);
  }

  template <int dim> 
  inline
  Tensor<1,dim> get_v (unsigned int q, std::vector<Vector<double> > old_solution_values)
  {
    Tensor<1,dim> v;      
    v[0] = old_solution_values[q](0);
    v[1] = old_solution_values[q](1);
      
    return v;    
  }

  template <int dim> 
  inline
  Tensor<1,dim> get_v_LinV (const Tensor<1,dim> phi_i_v)
  {
    Tensor<1,dim> tmp;
    tmp[0] = phi_i_v[0];
    tmp[1] = phi_i_v[1];
     
    return tmp;    
  }

  template <int dim> 
  inline
  Tensor<1,dim> get_u (unsigned int q, std::vector<Vector<double> > old_solution_values)
  {
    Tensor<1,dim> u;     
    u[0] = old_solution_values[q](dim);
    u[1] = old_solution_values[q](dim+1);
     
    return u;          
  }

  template <int dim> 
  inline
  Tensor<1,dim> get_u_LinU (const Tensor<1,dim> phi_i_u)
  {
    Tensor<1,dim> tmp;     
    tmp[0] = phi_i_u[0];
    tmp[1] = phi_i_u[1];
     
    return tmp;    
  }

  template <int dim> 
  inline
  double get_J_LinU (unsigned int q, const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads, const Tensor<2,dim> phi_i_grads_u)     
  {
    return (phi_i_grads_u[0][0] * (1 + old_solution_grads[q][dim+1][1]) +
         (1 + old_solution_grads[q][dim][0]) * phi_i_grads_u[1][1] -
         phi_i_grads_u[0][1] * old_solution_grads[q][dim+1][0] - 
         old_solution_grads[q][dim][1] * phi_i_grads_u[1][0]);  
  }

  template <int dim> 
  inline
  double get_J_Inverse_LinU (const double J, const double J_LinU)
  {
    return (-1.0/std::pow(J,2) * J_LinU);
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_F_LinU (const Tensor<2,dim> phi_i_grads_u)  
  {
    Tensor<2,dim> tmp;
    tmp[0][0] = phi_i_grads_u[0][0];
    tmp[0][1] = phi_i_grads_u[0][1];
    tmp[1][0] = phi_i_grads_u[1][0];
    tmp[1][1] = phi_i_grads_u[1][1];
    
    return tmp;
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_F_Inverse_LinU (const Tensor<2,dim> phi_i_grads_u,
           const double J,
           const double J_LinU,
           unsigned int q,
           std::vector<std::vector<Tensor<1,dim> > > old_solution_grads
           )  
  {
    Tensor<2,dim> F_tilde;
    F_tilde[0][0] = 1.0 + old_solution_grads[q][dim+1][1];
    F_tilde[0][1] = -old_solution_grads[q][dim][1];
    F_tilde[1][0] = -old_solution_grads[q][dim+1][0];
    F_tilde[1][1] = 1.0 + old_solution_grads[q][dim][0];
    
    Tensor<2,dim> F_tilde_LinU;
    F_tilde_LinU[0][0] = phi_i_grads_u[1][1];
    F_tilde_LinU[0][1] = -phi_i_grads_u[0][1];
    F_tilde_LinU[1][0] = -phi_i_grads_u[1][0];
    F_tilde_LinU[1][1] = phi_i_grads_u[0][0];

    return (-1.0/(J*J) * J_LinU * F_tilde +
      1.0/J * F_tilde_LinU);
 
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_J_F_Inverse_T_LinU (const Tensor<2,dim> phi_i_grads_u)  
  {
    Tensor<2,dim> tmp;
    tmp[0][0] = phi_i_grads_u[1][1];
    tmp[0][1] = -phi_i_grads_u[1][0];
    tmp[1][0] = -phi_i_grads_u[0][1];
    tmp[1][1] = phi_i_grads_u[0][0];
     
    return  tmp;
  }


  template <int dim> 
  inline
  double get_tr_C_LinU (unsigned int q, const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
     const Tensor<2,dim> phi_i_grads_u)     
  {
    return ((1 + old_solution_grads[q][dim][0]) *
      phi_i_grads_u[0][0] + 
      old_solution_grads[q][dim][1] *
      phi_i_grads_u[0][1] +
      (1 + old_solution_grads[q][dim+1][1]) *
      phi_i_grads_u[1][1] + 
      old_solution_grads[q][dim+1][0] *
      phi_i_grads_u[1][0]);
  }
}

// Second, we define the ALE transformations rules. These
// are used to transform the fluid equations from the Eulerian
// coordinate system to an arbitrary fixed reference 
// configuration.
namespace NSE_in_ALE
{
  template <int dim> 
  inline
  Tensor<2,dim> get_stress_fluid_ALE (const double density,
           const double viscosity,  
           const Tensor<2,dim>  pI,
           const Tensor<2,dim>  grad_v,
           const Tensor<2,dim>  grad_v_T,
           const Tensor<2,dim>  F_Inverse,
           const Tensor<2,dim>  F_Inverse_T)
  {    
    return (-pI + density * viscosity * (grad_v * F_Inverse + F_Inverse_T * grad_v_T ));
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_stress_fluid_except_pressure_ALE (const double density,
          const double viscosity, 
          const Tensor<2,dim>  grad_v,
          const Tensor<2,dim>  grad_v_T,
          const Tensor<2,dim>  F_Inverse,
          const Tensor<2,dim>  F_Inverse_T)
  {
    return (density * viscosity * (grad_v * F_Inverse + F_Inverse_T * grad_v_T));
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_stress_fluid_ALE_1st_term_LinAll (const Tensor<2,dim>  pI,
          const Tensor<2,dim>  F_Inverse_T,
          const Tensor<2,dim>  J_F_Inverse_T_LinU,              
          const Tensor<2,dim>  pI_LinP,
          const double J)
  {          
    return (-J * pI_LinP * F_Inverse_T - pI * J_F_Inverse_T_LinU);       
  }
  
  template <int dim> 
  inline
  Tensor<2,dim> get_stress_fluid_ALE_2nd_term_LinAll_short (const Tensor<2,dim> J_F_Inverse_T_LinU,             
                const Tensor<2,dim> stress_fluid_ALE,
                const Tensor<2,dim> grad_v,
                const Tensor<2,dim> grad_v_LinV,              
                const Tensor<2,dim> F_Inverse,
                const Tensor<2,dim> F_Inverse_LinU,             
                const double J,
                const double viscosity,
                const double density 
                )  
  {
    Tensor<2,dim> sigma_LinV;
    Tensor<2,dim> sigma_LinU;

    sigma_LinV = grad_v_LinV * F_Inverse + transpose(F_Inverse) * transpose(grad_v_LinV);
    sigma_LinU = grad_v *  F_Inverse_LinU + transpose(F_Inverse_LinU) * transpose(grad_v);
 
    return (density * viscosity * 
      (sigma_LinV + sigma_LinU) * J * transpose(F_Inverse) +
      stress_fluid_ALE * J_F_Inverse_T_LinU);    
  }

  template <int dim> 
  inline
  Tensor<2,dim> get_stress_fluid_ALE_3rd_term_LinAll_short (const Tensor<2,dim> F_Inverse,         
                const Tensor<2,dim> F_Inverse_LinU,              
                const Tensor<2,dim> grad_v,
                const Tensor<2,dim> grad_v_LinV,              
                const double viscosity,
                const double density,
                const double J,
                const Tensor<2,dim> J_F_Inverse_T_LinU)                        
  {
    return density * viscosity * 
      (J_F_Inverse_T_LinU * transpose(grad_v) * transpose(F_Inverse) +
       J * transpose(F_Inverse) * transpose(grad_v_LinV) * transpose(F_Inverse) +
       J * transpose(F_Inverse) * transpose(grad_v) * transpose(F_Inverse_LinU));  
  }

  template <int dim> 
  inline
  double get_Incompressibility_ALE (unsigned int q,
           std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)   
  {
    return (old_solution_grads[q][0][0] + old_solution_grads[q][dim+1][1] * old_solution_grads[q][0][0] -
      old_solution_grads[q][dim][1] * old_solution_grads[q][1][0] - old_solution_grads[q][dim+1][0] * old_solution_grads[q][0][1] +
      old_solution_grads[q][1][1] + old_solution_grads[q][dim][0] * old_solution_grads[q][1][1]); 

  }

  template <int dim>
  inline
  double
  get_Incompressibility_ALE_LinAll (const Tensor<2,dim> phi_i_grads_v,
                    const Tensor<2,dim> phi_i_grads_u,
                    unsigned int q,
                    const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
  {
    return (phi_i_grads_v[0][0] + phi_i_grads_v[1][1] +
        phi_i_grads_u[1][1] * old_solution_grads[q][0][0] + old_solution_grads[q][dim+1][1] * phi_i_grads_v[0][0] -
        phi_i_grads_u[0][1] * old_solution_grads[q][1][0] - old_solution_grads[q][dim+0][1] * phi_i_grads_v[1][0] -
        phi_i_grads_u[1][0] * old_solution_grads[q][0][1] - old_solution_grads[q][dim+1][0] * phi_i_grads_v[0][1] +
        phi_i_grads_u[0][0] * old_solution_grads[q][1][1] + old_solution_grads[q][dim+0][0] * phi_i_grads_v[1][1]);
  } 

  template <int dim> 
  inline
  Tensor<1,dim> get_Convection_LinAll_short (const Tensor<2,dim> phi_i_grads_v,
             const Tensor<1,dim> phi_i_v,
             const double J,
             const double J_LinU,
             const Tensor<2,dim> F_Inverse,
             const Tensor<2,dim> F_Inverse_LinU,                
             const Tensor<1,dim> v,
             const Tensor<2,dim> grad_v,        
             const double density              
             )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)v = rho J grad(v)F^{-1}v
    
    Tensor<1,dim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * v +
           J * grad_v * F_Inverse_LinU * v);
    
    Tensor<1,dim> convection_LinV;
    convection_LinV = (J * (phi_i_grads_v * F_Inverse * v + 
          grad_v * F_Inverse * phi_i_v));
    
    return density * (convection_LinU + convection_LinV);
  }
  

  template <int dim> 
  inline
  Tensor<1,dim> get_Convection_u_LinAll_short (const Tensor<2,dim> phi_i_grads_v,
         const Tensor<1,dim> phi_i_u,
         const double J,
         const double J_LinU,         
         const Tensor<2,dim>  F_Inverse,
         const Tensor<2,dim>  F_Inverse_LinU,
         const Tensor<1,dim>  u,
         const Tensor<2,dim>  grad_v,       
         const double density              
         )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u
    
    Tensor<1,dim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * u +
           J * grad_v * F_Inverse_LinU * u +
           J * grad_v * F_Inverse * phi_i_u);
    
    Tensor<1,dim> convection_LinV;
    convection_LinV = (J * phi_i_grads_v * F_Inverse * u); 
        
    return density * (convection_LinU + convection_LinV);
  }


  
  template <int dim> 
  inline
  Tensor<1,dim> get_Convection_u_old_LinAll_short (const Tensor<2,dim> phi_i_grads_v,        
             const double J,
             const double J_LinU,        
             const Tensor<2,dim>  F_Inverse,
             const Tensor<2,dim>  F_Inverse_LinU,             
             const Tensor<1,dim>  old_timestep_solution_displacement, 
             const Tensor<2,dim>  grad_v,       
             const double density                              
             )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u
    
    Tensor<1,dim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * old_timestep_solution_displacement +
           J * grad_v * F_Inverse_LinU * old_timestep_solution_displacement);
    
    Tensor<1,dim> convection_LinV;
    convection_LinV = (J * phi_i_grads_v * F_Inverse * old_timestep_solution_displacement); 
    
    return density * (convection_LinU  + convection_LinV);
  }



  
  //Linearization convection term concentration sym( ALE( v_f * nabla c_f ) )
  template <int dim> 
  inline
  double get_Convection_c_LinAll_short (const Tensor<1,dim> phi_i_grads_c,
             const Tensor<1,dim> phi_i_v,
             const double J,
             const double J_LinU,
             const Tensor<2,dim> F_Inverse,
             const Tensor<2,dim> F_Inverse_LinU,                
             const Tensor<1,dim> v,
             const Tensor<1,dim> grad_co
             )
  {
    // Linearization of concentration fluid convection term
    // sym ( v_f F^(-1) nabla c_f J )
    
    double convection_c_LinU = 0;
    double convection_c_LinV = 0;
    double convection_c_LinC = 0;

    for( int i=0; i<dim; i++ )
    {
      for( int j=0; j<dim; j++ )
      {
        convection_c_LinU += J_LinU * grad_co[i] * F_Inverse[i][j] * v[j]
                            + J * grad_co[i] * F_Inverse_LinU[i][j] * v[j];
        convection_c_LinC += J * phi_i_grads_c[i] * F_Inverse[i][j] * v[j];
        convection_c_LinV += J * grad_co[i] * F_Inverse[i][j] * phi_i_v[j];
      }
    }
    
    return convection_c_LinU + convection_c_LinC + convection_c_LinV;
  }

  template <int dim> 
  inline
  double get_Convection_c_u_LinAll_short (const Tensor<1,dim> phi_i_grads_c,
         const Tensor<1,dim> phi_i_u,
         const double J,
         const double J_LinU,         
         const Tensor<2,dim>  F_Inverse,
         const Tensor<2,dim>  F_Inverse_LinU,
         const Tensor<1,dim>  u,
         const Tensor<1,dim>  grad_co        
         )
  {
    double convection_c_LinU = 0;
    double convection_c_LinC = 0;

    for( int i=0; i<dim; i++ )
    {
      for( int j=0; j<dim; j++ )
      {
        convection_c_LinU += J_LinU * grad_co[i] * F_Inverse[i][j] * u[j]
                            + J * grad_co[i] * F_Inverse_LinU[i][j] * u[j]
                            + J * grad_co[i] * F_Inverse[i][j] * phi_i_u[j];
        convection_c_LinC += J * phi_i_grads_c[i] * F_Inverse[i][j] * u[j];
      }
    }        
    return convection_c_LinU + convection_c_LinC;
  }

  template <int dim> 
  inline
  double get_Convection_c_u_old_LinAll_short (const Tensor<1,dim> phi_i_grads_c,        
             const double J,
             const double J_LinU,        
             const Tensor<2,dim>  F_Inverse,
             const Tensor<2,dim>  F_Inverse_LinU,             
             const Tensor<1,dim>  old_timestep_solution_displacement, 
             const Tensor<1,dim>  grad_co             
             )
  {
    double convection_c_LinU = 0;
    double convection_c_LinC = 0;

    for( int i=0; i<dim; i++ )
    {
      for( int j=0; j<dim; j++ )
      {
        convection_c_LinU += J_LinU * grad_co[i] * F_Inverse[i][j] * old_timestep_solution_displacement[j]
                            + J * grad_co[i] * F_Inverse_LinU[i][j] * old_timestep_solution_displacement[j];
        convection_c_LinC += J * phi_i_grads_c[i] * F_Inverse[i][j] * old_timestep_solution_displacement[j];
      }
    }
    return convection_c_LinU  + convection_c_LinC;
  }

  template <int dim> 
  inline
  Tensor<1,dim> get_diffusion_term_conc_LinAll (const Tensor<1,dim> phi_i_grads_c,        
             const double J,
             const double J_LinU,        
             const Tensor<2,dim>  F_Inverse,
             const Tensor<2,dim>  F_Inverse_LinU,
             const Tensor<2,dim>  F_Inverse_T,
             const Tensor<2,dim>  F_Inverse_T_LinU,
             const Tensor<1,dim>  grad_co,
             double Df
            )
  {
    Tensor<1,dim> diffusion_c_LinU = J_LinU * Df * grad_co * F_Inverse * F_Inverse_T
                                + J * Df * grad_co * F_Inverse_LinU * F_Inverse_T
                                + J * Df * grad_co * F_Inverse * F_Inverse_T_LinU;
    Tensor<1,dim> diffusion_c_LinC = J * Df * phi_i_grads_c * F_Inverse * F_Inverse_T;
    return diffusion_c_LinU + diffusion_c_LinC;
  }
  

  template <int dim> 
  inline
  Tensor<1,dim> 
  get_accelaration_term_LinAll (const Tensor<1,dim> phi_i_v,
              const Tensor<1,dim> v,
              const Tensor<1,dim> old_timestep_v,
              const double J_LinU,
              const double J,
              const double old_timestep_J,
              const double density)
  {   
    return density/2.0 * (J_LinU * (v - old_timestep_v) + (J + old_timestep_J) * phi_i_v);
  }

  template <int dim> 
  inline
  double
  get_accelaration_term_conc_LinAll (const double phi_i_c,
              const double conc,
              const double old_timestep_conc,
              const double J_LinU,
              const double J,
              const double old_timestep_J,
              const double theta)
  {   
    return J_LinU * 0.5 * (conc - old_timestep_conc) + (theta * J + (1-theta) * old_timestep_J) * phi_i_c;
    
  }
}


// In the third namespace, we summarize the 
// constitutive relations for the structure equations.
namespace Structure_Terms_in_ALE
{
  // Green-Lagrange strain tensor
  template <int dim> 
  inline
  Tensor<2,dim> get_E (const Tensor<2,dim> F_T, const Tensor<2,dim> F, 
     const Tensor<2,dim> Identity, const double g_growth)
  {    
    return 0.5 * (1.0/(g_growth * g_growth) * F_T * F - Identity);
  }

  template <int dim> 
  inline
  double get_tr_E (const Tensor<2,dim> E)
  {     
    return trace (E);
  }

  template <int dim> 
  inline
  double get_tr_E_LinU (unsigned int q, 
     const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
     const Tensor<2,dim> phi_i_grads_u)     
  {
    return ((1 + old_solution_grads[q][dim][0]) *
      phi_i_grads_u[0][0] + 
      old_solution_grads[q][dim][1] *
      phi_i_grads_u[0][1] +
      (1 + old_solution_grads[q][dim+1][1]) *
      phi_i_grads_u[1][1] + 
      old_solution_grads[q][dim+1][0] *
      phi_i_grads_u[1][0]); 
  }
}


template <int dim>
class BoundarySolid : public Function<dim> 
{
public:
  BoundarySolid ( )    
    : Function<dim>(dim+dim+1+1) 
  {

  }
    
  virtual double value (const Point<dim>   &p,
      const unsigned int  component = 0) const;
};

template <int dim>
double
BoundarySolid<dim>::value (const Point<dim>  &p,
           const unsigned int component) const
{
  Assert (component < this->n_components,
    ExcIndexRange (component, 0, this->n_components));
  if( component == 5 )
    return 1;
  return 0;
}


// In this class, we define a function
// that deals with the boundary values.
// For our configuration, 
// we impose of parabolic inflow profile for the
// velocity at the left hand side of the channel. We choose
// a time dependent inflow profile with smooth 
// increase, to avoid difficulties at the beginning
// of the computation.      
template <int dim>
class BoundaryParabel : public Function<dim> 
{
public:
  BoundaryParabel (double time,
       const double u_y,
       const double compute_short_scale)    
    : Function<dim>(dim+dim+1+1) 
  {
    _time = time;  
    _u_y = u_y;
    _compute_short_scale = compute_short_scale;
  }
    
  virtual double value (const Point<dim>   &p,
      const unsigned int  component = 0) const;

  virtual void vector_value (const Point<dim> &p, 
           Vector<double>   &value) const;

private:
  double _time;
  double _u_y;
  double _compute_short_scale;

};

// The boundary values are given to component 
// with number 0.
template <int dim>
double
BoundaryParabel<dim>::value (const Point<dim>  &p,
           const unsigned int component) const
{
  Assert (component < this->n_components,
    ExcIndexRange (component, 0, this->n_components));

  const long double pi = 3.141592653589793238462643;
  
  double inflow_velocity = 2.5*1e-03;
  double inflow = 2;


  double beta_inflow = 1;//1.0e-1; //0.1;

  if (component == 0 )
  {
    if (!_compute_short_scale)
    {
      return   ( (p(0) == -5) && (p(1) <= 1.0) && (p(1) >= -1.0) ? inflow_velocity * 
         ((beta_inflow + 10.0 * (1.0 -  _u_y)) * (1.0 - p(1)*p(1))) : 0);
    }
    else if (_compute_short_scale)
    {
      //return 0;
      double sin_tmp = (1.0 + std::sin(2.0*pi*_time));

      double total_inflow = ( (p(0) == -5) && (p(1) <= 1.0) && (p(1) >= -1.0) ?  inflow_velocity * 
         sin_tmp * 
         ((beta_inflow + 10.0 * (1.0 -  _u_y)) * (1.0 - p(1)*p(1))) : 0);

      return total_inflow;
    }
  }
  if (component == 5)   
  {
    if (!_compute_short_scale)
    {
      return   ( (p(0) == 5) && (p(1) <= 1.0) && (p(1) >= -1.0) ? 10 - inflow * 
         ((beta_inflow + 10.0 * (1.0 -  _u_y)) * (1.0 - p(1)*p(1))) : 5);
    }
    else if (_compute_short_scale)
    {
      double sin_tmp = (1.0 + std::sin(2.0*pi*_time));

      double total_inflow = ( (p(0) == 5) && (p(1) <= 1.0) && (p(1) >= -1.0) ? 2 - inflow * 
         sin_tmp * 
         ((beta_inflow + 10.0 * (1.0 -  _u_y)) * (1.0 - p(1)*p(1))) : 2);

      return total_inflow;
    }
  }
  return 0;
}



template <int dim>
void
BoundaryParabel<dim>::vector_value (const Point<dim> &p,
            Vector<double>   &values) const 
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values (c) = BoundaryParabel<dim>::value (p, c);
}


// In the next class, we define the main problem at hand.
// Here, we implement
// the top-level logic of solving a
// time dependent FSCI problem in a 
// monolithic ALE framework.

// To construct the ALE mapping for the fluid mesh motion, we 
// solve an additional partial differential equation that 
// is given by the biharmonic equation. This kind of equation
// will be split into two equations (Lit. Ciarlet), 
// to avoid H^2 conforming finite elements.
//
// All equations are written in a common global system that 
// is referred to as a monolithic solution algorithm.
// 
// The discretization of the continuous problem is organized
// as follows: 
// - time discretization is based on finite differences 
// - spatial discretization is based on a Galerkin finite element scheme
// - the non-linear problem is solved by a Newton-like method 
//
// The  program is organized as follows. First, we set up
// runtime parameters and the system as done in other deal.II tutorial steps. 
// Then, we assemble
// the system matrix (Jacobian of Newton's method) 
// and system right hand side (residual of Newton's method) for the non-linear
// system. Two functions for the boundary values are provided because
// we are only supposed to apply boundary values in the first Newton step. In the
// subsequent Newton steps all Dirichlet values have to be equal zero.
// Afterwards, the routines for solving the linear 
// system and the Newton iteration are self-explaining. The following
// function is standard in deal.II tutorial steps:
// writing the solutions to graphical output. 
// The last three functions provide the framework to compute 
// functional values of interest. For the given fluid-structure
// interaction problem, we compute the displacement in the x- and y-directions 
// of the structure at a certain point. We are also interested in the observation
// of the drag- and lift evaluations, which are achieved by line-integration over faces.    
template <int dim>
class FSI_ALE_Problem 
{
public:
  
  FSI_ALE_Problem (const unsigned int degree);
  ~FSI_ALE_Problem (); 
  void run ();
  
private:

  void set_runtime_parameters ();
  void setup_system ();
  void setup_timestep ();
  void assemble_system_matrix ();   
  void assemble_system_rhs ();

  void refine_grid();
  
  void set_initial_condition();
  void set_initial_bc (const double time);
  void set_newton_bc ();
  
  void solve ();
  void newton_iteration(const double time);       
  void output_results (const unsigned int refinement_cycle,
           const BlockVector<double> solution) const;
  
  double compute_point_value (Point<dim> p,
            const unsigned int component) const;
  
  void compute_outflow ();
  void compute_drag_lift_fsi_fluid_tensor();
  void update_uy();
  void compute_functional_values (); 
  void compute_vorticity();
  void compute_minimal_J();
  void compute_line_velocity( double x_line);

  int number_coefficients = dim+dim+1+1;

  const unsigned int   degree;
  
  Triangulation<dim>   triangulation;
  FESystem<dim>        fe;
  DoFHandler<dim>      dof_handler;

  AffineConstraints<double>    constraints;  
  
  BlockSparsityPattern      sparsity_pattern; 
  BlockSparseMatrix<double> system_matrix; 
  
  BlockVector<double> solution, newton_update, old_timestep_solution;
  BlockVector<double> system_rhs;
  
  TimerOutput         timer;
  
  // Global variables for timestepping scheme   
  unsigned int timestep_number;
  unsigned int max_no_timesteps, max_no_timesteps_short_scale;  
  double timestep, theta, time; 
  std::string time_stepping_scheme;
  std::string test_case;

  // Fluid parameters 
  double density_fluid, viscosity; 
  
  // Structure parameters
  double density_structure; 
  double lame_coefficient_mu, lame_coefficient_lambda, poisson_ratio_nu;  

  //Biofilm concentration
  double k, K, k1, K1;

  //nutrients
  double c_n;
  double c_growth;

  //adhesion, detachment
  double ka, kd;

  //Diffusion coefficients
  double Df, Ds;

  // Other parameters to control the fluid mesh motion 
  double cell_diameter;  
  double alpha_u, alpha_us;
  
  double pressure_fluid_x;
  double stop_growth;
  double growth_displacement;
  double compute_short_scale, u_y, drag_summed, final_drag_summed, gamma_zero, growth_initial, drag;
 
  // Neumann
  double neumann_pressure_x, neumann_pressure_y;

  SparseDirectUMFPACK A_direct;

  const long double pi = 3.141592653589793238462643;
};


// The constructor of this class is comparable 
// to other tutorials steps, e.g., step-22, and step-31. 
// We are going to use the following finite element discretization: 
// Q_2^c for the fluid, Q_2^c for the structure, P_1^dc for the pressure, 
// and Q_2^c for the additional displacement. 
template <int dim>
FSI_ALE_Problem<dim>::FSI_ALE_Problem (const unsigned int degree)
                :
                degree (degree),
    triangulation (Triangulation<dim>::maximum_smoothing),
                fe (FE_Q<dim>(degree+1), dim, //velocity                   
        FE_Q<dim>(degree+1), dim, //displacement
        FE_DGP<dim>(degree), 1, // pressure
        FE_Q<dim>(degree), 1), // concentration (scalar-valued)),      
                dof_handler (triangulation),
    timer (std::cout, TimerOutput::summary, TimerOutput::cpu_times)   
{}


// This is the standard destructor.
template <int dim>
FSI_ALE_Problem<dim>::~FSI_ALE_Problem () 
{}


// In this method, we set up runtime parameters that 
// could also come from a paramter file. We propose
// three different configurations FSI 1, FSI 2, and FSI 3.
// The reader is invited to change these values to obtain
// other results. 
template <int dim>
void FSI_ALE_Problem<dim>::set_runtime_parameters ()
{
  test_case = "Case 1";
  drag_summed = 0.0;
  final_drag_summed = 0.0;
  gamma_zero = 5.0e-7;
  growth_initial = 100.0;
  compute_short_scale = 0.0;
  stop_growth = 1.0e+12;
  pressure_fluid_x = 1.0;
  growth_displacement = 1.;

  density_fluid = 1.0;
  density_structure = 3*1e+02; 
  viscosity = 1.0;//0.3;  // 1.0
  lame_coefficient_mu = 250./7.;//1.0e+3;  // 1.0e+4
  //poisson_ratio_nu = 0.2; 
  
  lame_coefficient_lambda =  1000./7.;//4.0e+4; //(2 * poisson_ratio_nu * lame_coefficient_mu)/(1.0 - 2 * poisson_ratio_nu);

  // Diffusion parameters to control the fluid mesh motion
  // The higher these parameters the stiffer the fluid mesh.
  alpha_u  = 1.0e-5;
  alpha_us = 1.0;

  //Biofilm Concentration coefficients
  k = 3;//*1e-2;//*1e-2;//3*1e-2; //max Wachstumsgeschwindigkeit
  K = 3;//*1e-3; //Halb-Sättigungskonstante - Michaelis-Menten-Konstante 
  k1 = k;//1;//1e-4;
  K1 = K;

  k = 3;//*1e-2;//3*1e-2; //max Wachstumsgeschwindigkeit
  K = 3*1e-4; //Halb-Sättigungskonstante - Michaelis-Menten-Konstante 
  k1 = 1e-4;//1;//1e-4;
  K1 = 1e-5;


  //Nutrients
  c_n = 0.00001;
  c_growth = 1.44*1e-06; //13.3 * 1e-08;

  //adhesion, detachment
  ka = 1e-05;//-1e-02;//-1;//-5*1e-01;//-1e-02;//-1e-01; 
  kd = 0; //1e-09;//-1e-05; //6

  //Diffusion coefficients
  Df = 2.5 * 1e-06;
  Ds = 2.5 * 1e-09;

  //Neumann
  neumann_pressure_x    = 0.1;
  neumann_pressure_y    = 0;
  
  // Timestepping schemes
  //BE, CN, CN_shifted
  time_stepping_scheme = "BE";

  // Timestep size:
  // TODO
  timestep = 10;//43200.0; //86400.0;

  // Maximum number of timesteps:
  // FSI 1: 25 , T= 25   (timestep == 1.0)
  // FSI 2: 1500, T= 15  (timestep == 1.0e-2)
  // FSI 3: 10000, T= 10 (timestep == 1.0e-3)
  max_no_timesteps = 10;//15000;
  max_no_timesteps_short_scale = 50;
  
  // A variable to count the number of time steps
  timestep_number = 0;

  // Counts total time  
  time = 0;

  // backward Euler scheme 
  theta = 1.0;

  // In the following, we read a *.inp grid from a file.
  // The geometry information is based on the 
  // fluid-structure interaction benchmark problems 
  // (Lit. J. Hron, S. Turek, 2006)
  std::string grid_name;
  //grid_name  = "channel_growth_L_Shape_Jul_2025.inp";
  grid_name  = "channel_growth_Sep_2013.inp";
  
  GridIn<dim> grid_in;
  grid_in.attach_triangulation (triangulation);
  std::ifstream input_file(grid_name.c_str());      
  Assert (dim==2, ExcInternalError());
  grid_in.read_ucd (input_file); 
  
  triangulation.refine_global (2);
}

// This function is similar to many deal.II tutorial steps.
template <int dim>
void FSI_ALE_Problem<dim>::setup_system ()
{
  timer.enter_subsection("Setup system.");

  // We set runtime parameters to drive the problem.
  // These parameters could also be read from a parameter file that
  // can be handled by the ParameterHandler object (see step-19)
  set_runtime_parameters ();

  system_matrix.clear ();
  
  dof_handler.distribute_dofs (fe);  
  DoFRenumbering::Cuthill_McKee (dof_handler);

  // We are dealing with 8 components for this 
  // two-dimensional fluid-structure interacion problem
  // Precisely, we use:
  // velocity in x and y:                0
  // structure displacement in x and y:  1
  // scalar pressure field:              2
  // additional displacement in x and y: 3
  // scalar concentration field:         4
  std::vector<unsigned int> block_component (6,0);
  block_component[dim] = 1;                     //displacement
  block_component[dim+1] = 1;                   //displacement
  block_component[dim+dim] = 2;                 //pressure
  block_component[dim+dim+1] = 3;           //concentration
 
  DoFRenumbering::component_wise (dof_handler, block_component);

  {        
    constraints.clear ();
    set_newton_bc ();
  }
  constraints.close ();
  
  std::vector<long unsigned int> dofs_per_block (3);
  dofs_per_block = DoFTools::count_dofs_per_fe_block (dof_handler, block_component);  
  const unsigned int n_v = dofs_per_block[0],
    n_u = dofs_per_block[1],
    n_p = dofs_per_block[2],
    n_c = dofs_per_block[3];

  std::cout << "Cells:\t"
            << triangulation.n_active_cells()
            << std::endl      
            << "DoFs:\t"
            << dof_handler.n_dofs()
            << " (" << n_v << '+' << n_u << '+' << n_p << '+' << n_c << ')'
            << std::endl;
 {
    BlockDynamicSparsityPattern csp (4,4);

    csp.block(0,0).reinit (n_v, n_v);
    csp.block(0,1).reinit (n_v, n_u);
    csp.block(0,2).reinit (n_v, n_p);
    csp.block(0,3).reinit (n_v, n_c);
  
    csp.block(1,0).reinit (n_u, n_v);
    csp.block(1,1).reinit (n_u, n_u);
    csp.block(1,2).reinit (n_u, n_p);
    csp.block(1,3).reinit (n_u, n_c);
  
    csp.block(2,0).reinit (n_p, n_v);
    csp.block(2,1).reinit (n_p, n_u);
    csp.block(2,2).reinit (n_p, n_p);
    csp.block(2,3).reinit (n_p, n_c);

    csp.block(3,0).reinit (n_c, n_v);
    csp.block(3,1).reinit (n_c, n_u);
    csp.block(3,2).reinit (n_c, n_p);
    csp.block(3,3).reinit (n_c, n_c);
 
    csp.collect_sizes();    
  

    DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);

    sparsity_pattern.copy_from (csp);
  }
 
  system_matrix.reinit (sparsity_pattern);

  // Actual solution at time step n
  solution.reinit (4);
  solution.block(0).reinit (n_v);
  solution.block(1).reinit (n_u);
  solution.block(2).reinit (n_p);
  solution.block(3).reinit (n_c);
 
  solution.collect_sizes ();
 
  // Old timestep solution at time step n-1
  old_timestep_solution.reinit (4);
  old_timestep_solution.block(0).reinit (n_v);
  old_timestep_solution.block(1).reinit (n_u);
  old_timestep_solution.block(2).reinit (n_p);
  old_timestep_solution.block(3).reinit (n_c);
 
  old_timestep_solution.collect_sizes ();


  // Updates for Newton's method
  newton_update.reinit (4);
  newton_update.block(0).reinit (n_v);
  newton_update.block(1).reinit (n_u);
  newton_update.block(2).reinit (n_p);
  newton_update.block(3).reinit (n_c);
 
  newton_update.collect_sizes ();
 
  // Residual for  Newton's method
  system_rhs.reinit (4);
  system_rhs.block(0).reinit (n_v);
  system_rhs.block(1).reinit (n_u);
  system_rhs.block(2).reinit (n_p);
  system_rhs.block(3).reinit (n_c);

  system_rhs.collect_sizes ();

  timer.leave_subsection(); 
}

// This function is similar to many deal.II tutorial steps.
template <int dim>
void FSI_ALE_Problem<dim>::setup_timestep ()
{
  timer.enter_subsection("Setup timestep.");

  // We set runtime parameters to drive the problem.
  // These parameters could also be read from a parameter file that
  // can be handled by the ParameterHandler object (see step-19)
  //set_runtime_parameters ();

  system_matrix.clear ();

  dof_handler.distribute_dofs (fe);  //
  DoFRenumbering::Cuthill_McKee (dof_handler); //

  //output_results (timestep_number+210000,solution);
  //dof_handler.distribute_dofs (fe);
  //output_results (timestep_number+220000,solution);
  //DoFRenumbering::Cuthill_McKee (dof_handler);

  //output_results (timestep_number+20000,solution);

  // We are dealing with 8 components for this
  // two-dimensional fluid-structure interacion problem
  // Precisely, we use:
  // velocity in x and y:                0
  // structure displacement in x and y:  1
  // scalar pressure field:              2
  // additional displacement in x and y: 3
  // scalar concentration field:         4
  std::vector<unsigned int> block_component (6,0);
  block_component[dim] = 1;                     //displacement
  block_component[dim+1] = 1;                   //displacement
  block_component[dim+dim] = 2;                 //pressure
  block_component[dim+dim+1] = 3;           //concentration

  DoFRenumbering::component_wise (dof_handler, block_component);

  {
    constraints.clear ();
    set_newton_bc ();
    DoFTools::make_hanging_node_constraints (dof_handler,
               constraints);
  }
  constraints.close ();
  //output_results (timestep_number+30000,solution);

  std::vector<long unsigned int> dofs_per_block (3);
  dofs_per_block = DoFTools::count_dofs_per_fe_block (dof_handler, block_component);  
  
  const unsigned int n_v = dofs_per_block[0],
    n_u = dofs_per_block[1],
    n_p = dofs_per_block[2],
    n_c = dofs_per_block[3];

  std::cout << "Cells:\t"
            << triangulation.n_active_cells()
            << std::endl
            << "DoFs:\t"
            << dof_handler.n_dofs()
            << " (" << n_v << '+' << n_u << '+' << n_p << '+' << n_c << ')'
            << std::endl;


 //output_results (timestep_number+40000,solution);

 {
    BlockDynamicSparsityPattern csp (4,4);

    csp.block(0,0).reinit (n_v, n_v);
    csp.block(0,1).reinit (n_v, n_u);
    csp.block(0,2).reinit (n_v, n_p);
    csp.block(0,3).reinit (n_v, n_c);

    csp.block(1,0).reinit (n_u, n_v);
    csp.block(1,1).reinit (n_u, n_u);
    csp.block(1,2).reinit (n_u, n_p);
    csp.block(1,3).reinit (n_u, n_c);

    csp.block(2,0).reinit (n_p, n_v);
    csp.block(2,1).reinit (n_p, n_u);
    csp.block(2,2).reinit (n_p, n_p);
    csp.block(2,3).reinit (n_p, n_c);

    csp.block(3,0).reinit (n_c, n_v);
    csp.block(3,1).reinit (n_c, n_u);
    csp.block(3,2).reinit (n_c, n_p);
    csp.block(3,3).reinit (n_c, n_c);

    csp.collect_sizes();


    DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);

    sparsity_pattern.copy_from (csp);
  }

  system_matrix.reinit (sparsity_pattern);

  // Actual solution at time step n
  solution.reinit (4);
  solution.block(0).reinit (n_v);
  solution.block(1).reinit (n_u);
  solution.block(2).reinit (n_p);
  solution.block(3).reinit (n_c);
 
  solution.collect_sizes ();
 
  // Old timestep solution at time step n-1
  old_timestep_solution.reinit (4);
  old_timestep_solution.block(0).reinit (n_v);
  old_timestep_solution.block(1).reinit (n_u);
  old_timestep_solution.block(2).reinit (n_p);
  old_timestep_solution.block(3).reinit (n_c);
 
  old_timestep_solution.collect_sizes ();

  // Updates for Newton's method
  newton_update.reinit (4);
  newton_update.block(0).reinit (n_v);
  newton_update.block(1).reinit (n_u);
  newton_update.block(2).reinit (n_p);
  newton_update.block(3).reinit (n_c);

  newton_update.collect_sizes ();

  // Residual for  Newton's method
  system_rhs.reinit (4);
  system_rhs.block(0).reinit (n_v);
  system_rhs.block(1).reinit (n_u);
  system_rhs.block(2).reinit (n_p);
  system_rhs.block(3).reinit (n_c);

  system_rhs.collect_sizes ();

  timer.leave_subsection();
}


// In this function, we assemble the Jacobian matrix
// for the Newton iteration. The fluid and the structure 
// equations are computed on different sub-domains
// in the mesh and ask for the corresponding 
// material ids. The fluid equations are defined on 
// mesh cells with the material id == 0 and the structure
// equations on cells with the material id == 1. 
//
// To compensate the well-known problem in fluid
// dynamics on the outflow boundary, we also
// add some correction term on the outflow boundary.
// This relation is known as `do-nothing' condition.
// In the inner loops of the local_cell_matrix, the 
// time dependent equations are discretized with
// a finite difference scheme. 
// Quasi-stationary processes (FSI 1) can be computed 
// by the BE scheme. The other two schemes are useful 
// for non-stationary computations (FSI 2 and FSI 3).
//
// Assembling of the inner most loop is treated with help of 
// the fe.system_to_component_index(j).first function from
// the library. 
// Using this function makes the assembling process much faster
// than running over all local degrees of freedom. 
template <int dim>
void FSI_ALE_Problem<dim>::assemble_system_matrix ()
{
  timer.enter_subsection("Assemble Matrix.");
  system_matrix=0;
     
  QGauss<dim>   quadrature_formula(degree+2);  
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);

  FESubfaceValues<dim> fe_face_values_neighbor( fe,
                                          face_quadrature_formula,
                                          update_values         | update_quadrature_points  |
                                          update_normal_vectors | update_gradients |
                                          update_JxW_values);
  
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
            update_values         | update_quadrature_points  |
            update_normal_vectors | update_gradients |
            update_JxW_values);
   
  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
  
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<long unsigned int> local_dof_indices (dofs_per_cell); 
    

  // Now, we are going to use the 
  // FEValuesExtractors to determine
  // the five principle variables
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Vector displacements (dim); // 2
  const FEValuesExtractors::Scalar pressure (dim+dim); // 4
  const FEValuesExtractors::Scalar concentration (dim+dim+1); // 5
 

  // We declare Vectors and Tensors for 
  // the solutions at the previous Newton iteration:
  std::vector<Vector<double> > old_solution_values (n_q_points, 
                Vector<double>(number_coefficients));

  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, 
                std::vector<Tensor<1,dim> > (number_coefficients));

  std::vector<Vector<double> >  old_solution_face_values (n_face_q_points, 
                Vector<double>(number_coefficients));
       
  std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points, 
                     std::vector<Tensor<1,dim> > (number_coefficients));
    
  // We declare Vectors and Tensors for 
  // the solution at the previous time step:
   std::vector<Vector<double> > old_timestep_solution_values (n_q_points, 
                Vector<double>(number_coefficients));


  std::vector<std::vector<Tensor<1,dim> > > old_timestep_solution_grads (n_q_points, 
              std::vector<Tensor<1,dim> > (number_coefficients));


  std::vector<Vector<double> >   old_timestep_solution_face_values (n_face_q_points, 
                    Vector<double>(number_coefficients));

    std::vector<Vector<double> >   old_timestep_solution_values_inv (n_face_q_points, 
                    Vector<double>(number_coefficients));
  
    
  std::vector<std::vector<Tensor<1,dim> > >  old_timestep_solution_face_grads (n_face_q_points, 
                         std::vector<Tensor<1,dim> > (number_coefficients));
   
  // Declaring test functions:
  std::vector<Tensor<1,dim> > phi_i_v (dofs_per_cell); 
  std::vector<Tensor<2,dim> > phi_i_grads_v(dofs_per_cell);
  std::vector<double>         phi_i_p(dofs_per_cell);   
  std::vector<Tensor<1,dim> > phi_i_u (dofs_per_cell); 
  std::vector<Tensor<2,dim> > phi_i_grads_u(dofs_per_cell);
  std::vector<double>         phi_i_c(dofs_per_cell);   
  std::vector<Tensor<1,dim> > phi_i_grads_c (dofs_per_cell);   
  std::vector<double>         phi_i_c_inv(fe.n_dofs_per_face());   

  // This is the identity matrix in two dimensions:
  const Tensor<2,dim> Identity = ALE_Transformations
    ::get_Identity<dim> ();
                       
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  bool debug = true;

  double interface_check;
  
  for (; cell!=endc; ++cell)
  { 
    fe_values.reinit (cell);
    local_matrix = 0;
      
    // We need the cell diameter to control the fluid mesh motion
    cell_diameter = cell->diameter();
      
    // Old Newton iteration values
    fe_values.get_function_values (solution, old_solution_values);
    fe_values.get_function_gradients (solution, old_solution_grads);
      
    // Old_timestep_solution values
    fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
    fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);
    
    // Next, we run over all cells for the fluid equations
    if (cell->material_id() == 0)
    {
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
          phi_i_v[k]       = fe_values[velocities].value (k, q);
          phi_i_grads_v[k] = fe_values[velocities].gradient (k, q);
          phi_i_p[k]       = fe_values[pressure].value (k, q);                   
          phi_i_u[k]       = fe_values[displacements].value (k, q);
          phi_i_grads_u[k] = fe_values[displacements].gradient (k, q);
          phi_i_c[k]       = fe_values[concentration].value (k, q);
          phi_i_grads_c[k] = fe_values[concentration].gradient (k, q);
        }

        int is_on_b = 0;
        interface_check = fe_values.get_quadrature().point(q)[1];

        if( interface_check - 0.887298 < 1e-06 && interface_check - 0.887298 > -1e-06 )
        {
          is_on_b = 1;
        }
        else if (  interface_check - 0.112702 < 1e-06 && interface_check - 0.112702 > -1e-06 )
        {
          is_on_b = 1;
        } //adhesion/attachement

        // We build values, vectors, and tensors
        // from information of the previous Newton step. These are introduced 
        // for two reasons:
        // First, these are used to perform the ALE mapping of the 
        // fluid equations. Second, these terms are used to 
        // make the notation as simple and self-explaining as possible:
        const double co = ALE_Transformations::get_co<dim> (q, old_solution_values);
        const Tensor<2,dim> pI = ALE_Transformations::get_pI<dim> (q, old_solution_values);
        const Tensor<1,dim> v = ALE_Transformations::get_v<dim> (q, old_solution_values);
        const Tensor<1,dim> u = ALE_Transformations::get_u<dim> (q,old_solution_values);
        const Tensor<2,dim> grad_u = ALE_Transformations ::get_grad_u<dim> (q, old_solution_grads);
        const Tensor<2,dim> grad_v = ALE_Transformations::get_grad_v<dim> (q, old_solution_grads);  
        const Tensor<1,dim> grad_co = ALE_Transformations::get_grad_co<dim> (q, old_solution_grads);
        const Tensor<2,dim> grad_v_T = ALE_Transformations::get_grad_v_T<dim> (grad_v);
        const Tensor<2,dim> F = ALE_Transformations::get_F<dim> (q, old_solution_grads);     
        const Tensor<2,dim> F_Inverse = ALE_Transformations::get_F_Inverse<dim> (F);
        const Tensor<2,dim> F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dim> (F_Inverse);
        const double J = ALE_Transformations::get_J<dim> (F);
        //double co_inv = 0; //adhesion/attachement
        
        // Stress tensor for the fluid in ALE notation        
        const Tensor<2,dim> sigma_ALE = NSE_in_ALE::get_stress_fluid_ALE<dim> (density_fluid, viscosity, pI, grad_v, grad_v_T, F_Inverse, F_Inverse_T );
                
        // Further, we also need some information from the previous time steps
        const Tensor<1,dim> old_timestep_v = ALE_Transformations::get_v<dim> (q, old_timestep_solution_values);
        const Tensor<1,dim> old_timestep_u = ALE_Transformations::get_u<dim> (q, old_timestep_solution_values);
        const Tensor<2,dim> old_timestep_F = ALE_Transformations::get_F<dim> (q, old_timestep_solution_grads);
        const double old_timestep_co = ALE_Transformations::get_co<dim> (q, old_timestep_solution_values);
        const double old_timestep_J = ALE_Transformations::get_J<dim> (old_timestep_F);
        //Ende Loesungskomponenten in Newton aus vorheriger Iteration

        //Richtungsableitungen
        // Outer loop for dofs
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        { 
          const Tensor<2,dim> pI_LinP = ALE_Transformations::get_pI_LinP<dim> (phi_i_p[i]);
          const Tensor<2,dim> grad_v_LinV = ALE_Transformations::get_grad_v_LinV<dim> (phi_i_grads_v[i]);
          const double J_LinU =  ALE_Transformations::get_J_LinU<dim> (q, old_solution_grads, phi_i_grads_u[i]);
          const Tensor<2,dim> J_F_Inverse_T_LinU = ALE_Transformations::get_J_F_Inverse_T_LinU<dim> (phi_i_grads_u[i]);
          const Tensor<2,dim> F_Inverse_LinU = ALE_Transformations::get_F_Inverse_LinU (phi_i_grads_u[i], J, J_LinU, q, old_solution_grads);
          const Tensor<2,dim> F_Inverse_T_LinU = transpose(F_Inverse_LinU);
          const Tensor<2,dim> stress_fluid_ALE_1st_term_LinAll = NSE_in_ALE::get_stress_fluid_ALE_1st_term_LinAll<dim> (pI, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP, J);
          const Tensor<2,dim> stress_fluid_ALE_2nd_term_LinAll = NSE_in_ALE::get_stress_fluid_ALE_2nd_term_LinAll_short 
                                                                    (J_F_Inverse_T_LinU, sigma_ALE, grad_v, grad_v_LinV, F_Inverse, F_Inverse_LinU, J, viscosity, density_fluid);  
          const Tensor<1,dim> convection_fluid_LinAll_short = NSE_in_ALE::get_Convection_LinAll_short<dim> 
                                                                  (phi_i_grads_v[i], phi_i_v[i], J,J_LinU, F_Inverse, F_Inverse_LinU, v, grad_v, density_fluid);
          const double incompressibility_ALE_LinAll = NSE_in_ALE::get_Incompressibility_ALE_LinAll<dim> (phi_i_grads_v[i], phi_i_grads_u[i], q, old_solution_grads); 
          const Tensor<1,dim> accelaration_term_LinAll = NSE_in_ALE::get_accelaration_term_LinAll (phi_i_v[i], v, old_timestep_v, J_LinU, J, old_timestep_J, density_fluid);
          const Tensor<1,dim> convection_fluid_u_LinAll_short =  NSE_in_ALE::get_Convection_u_LinAll_short<dim> (phi_i_grads_v[i], phi_i_u[i], J,J_LinU, F_Inverse, F_Inverse_LinU, u, grad_v, density_fluid);
          const Tensor<1,dim> convection_fluid_u_old_LinAll_short = NSE_in_ALE::get_Convection_u_old_LinAll_short<dim> (phi_i_grads_v[i], J, J_LinU, F_Inverse, F_Inverse_LinU, old_timestep_u, grad_v, density_fluid);

          /**********
           * Concentration
           **********/
          //Concentration Convection Term Linearized sym((v_f-w)F^-1 * nabla c_f J phi_c)
          const double convection_fluid_c_LinAll_short = NSE_in_ALE::get_Convection_c_LinAll_short<dim> (phi_i_grads_c[i], phi_i_v[i], J, J_LinU, F_Inverse, F_Inverse_LinU, v, grad_co); 
          const double convection_fluid_c_u_LinAll_short =  NSE_in_ALE::get_Convection_c_u_LinAll_short<dim> (phi_i_grads_c[i], phi_i_u[i], J,J_LinU, F_Inverse, F_Inverse_LinU, u, grad_co);
          const double convection_fluid_c_u_old_LinAll_short = NSE_in_ALE::get_Convection_c_u_old_LinAll_short<dim> (phi_i_grads_c[i], J, J_LinU, F_Inverse, F_Inverse_LinU, old_timestep_u, grad_co);
          
          //Concentration Accelaration Term Linearized: sym(partial_t c_f phi J)
          const double accelaration_term_conc_LinAll = NSE_in_ALE::get_accelaration_term_conc_LinAll<dim> (phi_i_c[i], co, old_timestep_co, J_LinU, J, old_timestep_J, theta);
          
          //sym+( J F^(-1) D_f nabla c_f  F^(-T))
          const Tensor<1,dim> diffusion_term_conc_LinAll = NSE_in_ALE::get_diffusion_term_conc_LinAll<dim> (phi_i_grads_c[i], J, J_LinU, F_Inverse, F_Inverse_LinU, F_Inverse_T, F_Inverse_T_LinU, grad_co, Df);

          //Schleife der Testfunktion
          // Inner loop for dofs
          for (unsigned int j=0; j<dofs_per_cell; ++j)
          { 
            // Fluid , NSE in ALE
            const unsigned int comp_j = fe.system_to_component_index(j).first; 
            if (comp_j == 0 || comp_j == 1)
            {   
              local_matrix(j,i) += (compute_short_scale * accelaration_term_LinAll * phi_i_v[j] +   
                                    + timestep * theta *            
                                    convection_fluid_LinAll_short * phi_i_v[j]              
                                    - convection_fluid_u_LinAll_short * phi_i_v[j]
                                    + convection_fluid_u_old_LinAll_short * phi_i_v[j]
                                    + timestep * scalar_product(stress_fluid_ALE_1st_term_LinAll, phi_i_grads_v[j])
                                    + timestep * theta *
                                    scalar_product(stress_fluid_ALE_2nd_term_LinAll, phi_i_grads_v[j])
                                    ) * fe_values.JxW(q);
            }             
            else if (comp_j == 2 || comp_j == 3)
            {
              local_matrix(j,i) += (-alpha_u/(J*J) * J_LinU * scalar_product(grad_u, phi_i_grads_u[j]) 
                                    + alpha_u/J * scalar_product(phi_i_grads_u[i], phi_i_grads_u[j])
                                        ) * fe_values.JxW(q);
            }
            else if (comp_j == 4)
            {
              local_matrix(j,i) += (incompressibility_ALE_LinAll *  phi_i_p[j] 
                                    ) * fe_values.JxW(q);   
            }
            else if (comp_j == 5)
            {
              local_matrix(j,i) += ( compute_short_scale * accelaration_term_conc_LinAll * phi_i_c[j]
                                      + timestep * theta * diffusion_term_conc_LinAll * phi_i_grads_c[j]
                                      + timestep * theta * convection_fluid_c_LinAll_short * phi_i_c[j]
                                      - convection_fluid_c_u_LinAll_short * phi_i_c[j]      //*timestep * 1./timestep
                                      + convection_fluid_c_u_old_LinAll_short * phi_i_c[j]  //*timestep * 1./timestep
                                      + timestep * theta * is_on_b * J * ( ka * phi_i_c[i]) * phi_i_c[j]
                                    ) * fe_values.JxW(q); 
            }
            // end j dofs  
          }   
          // end i dofs   
        }   
        // end n_q_points
      }
        
      // We compute in the following
      // one term on the outflow boundary. 
      // This relation is well-know in the literature 
      // as "do-nothing" condition. Therefore, we only
      // ask for the corresponding color at the outflow 
      // boundary that is 1 in our case.
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() &&      
            ((cell->face(face)->boundary_id() == 1)  ||
            (cell->face(face)->boundary_id() == 0))
           )
        {
          fe_face_values.reinit (cell, face);
          
          fe_face_values.get_function_values (solution, old_solution_face_values);
          fe_face_values.get_function_gradients (solution, old_solution_face_grads);  
          
          for (unsigned int q=0; q<n_face_q_points; ++q)
          {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
              phi_i_v[k]       = fe_face_values[velocities].value (k, q);
              phi_i_grads_v[k] = fe_face_values[velocities].gradient (k, q);    
              phi_i_grads_u[k] = fe_face_values[displacements].gradient (k, q);
            }
            const Tensor<2,dim>  grad_v = ALE_Transformations::get_grad_v<dim> (q, old_solution_face_grads);
            const Tensor<2,dim> F = ALE_Transformations::get_F<dim> (q, old_solution_face_grads);
            const Tensor<2,dim> F_Inverse = ALE_Transformations::get_F_Inverse<dim> (F);
            const double J = ALE_Transformations::get_J<dim> (F);
              
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const Tensor<2,dim> grad_v_LinV = ALE_Transformations::get_grad_v_LinV<dim> (phi_i_grads_v[i]);
              const double J_LinU = ALE_Transformations::get_J_LinU<dim> (q, old_solution_face_grads, phi_i_grads_u[i]);
              const Tensor<2,dim> J_F_Inverse_T_LinU = ALE_Transformations::get_J_F_Inverse_T_LinU<dim> (phi_i_grads_u[i]);
              const Tensor<2,dim> F_Inverse_LinU = ALE_Transformations::get_F_Inverse_LinU (phi_i_grads_u[i], J, J_LinU, q, old_solution_face_grads);
              const Tensor<2,dim> stress_fluid_ALE_3rd_term_LinAll =  NSE_in_ALE::get_stress_fluid_ALE_3rd_term_LinAll_short<dim> 
                                                          (F_Inverse, F_Inverse_LinU, grad_v, grad_v_LinV, viscosity, density_fluid, J, J_F_Inverse_T_LinU);
              
              // Here, we multiply the symmetric part of fluid's stress tensor
              // with the normal direction.
              const Tensor<1,dim> neumann_value = (stress_fluid_ALE_3rd_term_LinAll * fe_face_values.normal_vector(q));
                
              for (unsigned int j=0; j<dofs_per_cell; ++j)
              {        
                const unsigned int comp_j = fe.system_to_component_index(j).first; 
                if (comp_j == 0 || comp_j == 1 )
                {
                  local_matrix(j,i) -= (timestep * theta *
                                      neumann_value * phi_i_v[j] 
                                      ) * fe_face_values.JxW(q);
                }
                // end j    
              } 
              // end i
            }   
            // end q_face_points
          } 
          // end if-routine face integrals
        }         
        // end face integrals do-nothing
      }   
    

      // This is the same as discussed in step-22:
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (local_matrix, local_dof_indices,
              system_matrix);
    
      // Finally, we arrive at the end for assembling the matrix
      // for the fluid equations and step to the computation of the 
      // structure terms:
    } 
    else if (cell->material_id() == 1)
    {       
      for (unsigned int q=0; q<n_q_points; ++q)
      {       
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
          phi_i_v[k]       = fe_values[velocities].value (k, q);
          phi_i_grads_v[k] = fe_values[velocities].gradient (k, q);
          phi_i_p[k]       = fe_values[pressure].value (k, q);                   
          phi_i_u[k]       = fe_values[displacements].value (k, q);
          phi_i_grads_u[k] = fe_values[displacements].gradient (k, q);
          phi_i_c[k]       = fe_values[concentration].value (k, q);
          phi_i_grads_c[k] = fe_values[concentration].gradient (k, q);
        }
       
        int is_on_b = 0;
        interface_check = fe_values.get_quadrature().point(q)[1];
        if( interface_check - 0.887298 < 1e-06 && interface_check - 0.887298 > -1e-06 )
        {
          is_on_b = 1;
        }
        else if (  interface_check - 0.112702 < 1e-06 && interface_check - 0.112702 > -1e-06 )
        {
          is_on_b = 1;
        }

        // It is here the same as already shown for the fluid equations.
        // First, we prepare things coming from the previous Newton
        // iteration...
        const double co = ALE_Transformations::get_co<dim> (q, old_solution_values);
        const Tensor<1,dim> grad_co = ALE_Transformations::get_grad_co<dim> (q, old_solution_grads);
        const Tensor<1,dim> v = ALE_Transformations::get_v<dim> (q, old_solution_values);
        const Tensor<1,dim> u = ALE_Transformations::get_u<dim> (q, old_solution_values);
        const Tensor<2,dim> F = ALE_Transformations::get_F<dim> (q, old_solution_grads);
        const Tensor<2,dim> F_T = ALE_Transformations::get_F_T<dim> (F);
        const double J = ALE_Transformations::get_J<dim> (F);
                
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          //TODO linCo
          const double J_LinU = ALE_Transformations::get_J_LinU<dim> (q, old_solution_grads, phi_i_grads_u[i]);
          const Tensor<2,dim> F_LinU = ALE_Transformations::get_F_LinU<dim> (phi_i_grads_u[i]);
          const Tensor<2,dim> F_Inverse_LinU = ALE_Transformations::get_F_Inverse_LinU<dim> (phi_i_grads_u[i], J, J_LinU, q, old_solution_grads);
          const Tensor<2,dim> F_Inverse_T_LinU = transpose(F_Inverse_LinU);
          double old_timestep_co = ALE_Transformations::get_co<dim> (q, old_timestep_solution_values); 
          const Tensor<1,dim> old_timestep_u = ALE_Transformations::get_u<dim> (q, old_timestep_solution_values);  

          // Wachstum
          double g_growth = 0.0;
          if ((std::abs(fe_values.quadrature_point(q)[1]) > 1.0) && fe_values.quadrature_point(q)[1] > 0 && (time <=stop_growth))
          {
            g_growth = 1.0 + 1e-02 * old_timestep_co *
                      std::exp(-(fe_values.quadrature_point(q)[0]-growth_displacement) * (fe_values.quadrature_point(q)[0]- growth_displacement))
                      * (2.0 - std::abs(fe_values.quadrature_point(q)[1]));
          }
          else if ((std::abs(fe_values.quadrature_point(q)[1]) > 1.0) && fe_values.quadrature_point(q)[1] < 0 && (time <=stop_growth))
          {
            g_growth = 1.0 + 1e-02 * old_timestep_co *
                      std::exp(-(fe_values.quadrature_point(q)[0]+growth_displacement) * (fe_values.quadrature_point(q)[0]+ growth_displacement))
                      * (2.0 - std::abs(fe_values.quadrature_point(q)[1]));
          }
          else if ((std::abs(fe_values.quadrature_point(q)[1]) > 1.0) && (time > stop_growth))
          {
            abort();
          }
          else 
            g_growth = 1.0;

          // STVK: Green-Lagrange strain tensor derivatives
          const Tensor<2,dim> E = Structure_Terms_in_ALE ::get_E<dim> (F_T, F, Identity, g_growth);
          const double tr_E = Structure_Terms_in_ALE::get_tr_E<dim> (E);
          
          const Tensor<2,dim> E_Lin_All = 0.5 * 1.0/(g_growth * g_growth) * (transpose(F_LinU) * F + transpose(F) * F_LinU);
          const double tr_E_Lin_All = trace(E_Lin_All);
          
          //sym( lambda * tr(E) * I + 2mu * E + R_s (c) * I )
          Tensor<2,dim> sigma_co_lin = lame_coefficient_lambda * tr_E_Lin_All * Identity
                                  + 2 * lame_coefficient_mu * E_Lin_All ;
        

          for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
            // STVK 
            const unsigned int comp_j = fe.system_to_component_index(j).first; 
            if (comp_j == 0 || comp_j == 1)
            {
              local_matrix(j,i) += (density_structure * phi_i_v[i] * phi_i_v[j]                  
                                    + timestep * theta * scalar_product( sigma_co_lin,
                                    phi_i_grads_v[j]) 
                                    ) * fe_values.JxW(q);       
            }        
            else if (comp_j == 2 || comp_j == 3)
            {
              local_matrix(j,i) +=  (density_structure * alpha_us * 
                                    (phi_i_u[i] * phi_i_u[j] 
                                      - timestep * theta * phi_i_v[i] * phi_i_u[j])           
                                    ) *  fe_values.JxW(q); 
            }
            else if (comp_j == 4)
            {
              local_matrix(j,i) += (phi_i_p[i] * phi_i_p[j]) * fe_values.JxW(q);      
            }
            else if (comp_j == 5)
            {
              const Tensor<1, dim> grad_phi_i_c = fe_values[concentration].gradient(i, q);
              const double phi_j_c = fe_values[concentration].value(j, q);
              //const Tensor<1, dim> grad_phi_j_c = fe_values[concentration].gradient(j, q);

              local_matrix(j,i) += (compute_short_scale * phi_i_c[i] * phi_i_c[j]
                                    - timestep * theta * (grad_co * phi_i_v[i] + phi_i_grads_c[i] * v ) * phi_i_c[j]
                                    - timestep * theta * (phi_i_c[i] * v + co * phi_i_v[i] ) * phi_i_grads_c[j]
                                    + timestep * ( Ds * phi_i_grads_c[i] ) * phi_i_grads_c[j] 
                                    //- timestep * ( k * c_n )/(K + c_n) * phi_i_c[i] * phi_i_c[j]
                                    + timestep * theta * is_on_b * ( kd * phi_i_c[i]) * phi_i_c[j] 
              ) * fe_values.JxW(q); 
            }
          } // end j dofs
        } // end i dofs 
        debug = false;
      } // end n_q_points 

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (local_matrix, local_dof_indices,
                system_matrix);
    } // end if (second PDE: STVK material)
  } // end cell
  timer.leave_subsection(); 
}


// In this function we assemble the semi-linear 
// of the right hand side of Newton's method (its residual).
// The framework is in principal the same as for the 
// system matrix.
template <int dim>
void
FSI_ALE_Problem<dim>::assemble_system_rhs ()
{
  timer.enter_subsection("Assemble Rhs.");
  system_rhs=0;
  
  QGauss<dim>   quadrature_formula(degree+2);
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);

  FESubfaceValues<dim> fe_face_values_neighbor( fe,
                                          face_quadrature_formula,
                                          update_values         | update_quadrature_points  |
                                          update_normal_vectors | update_gradients |
                                          update_JxW_values);

  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
            update_values         | update_quadrature_points  |
            update_normal_vectors | update_gradients |
            update_JxW_values);

  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
  
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();
 
  Vector<double>       local_rhs (dofs_per_cell);

  std::vector<long unsigned int> local_dof_indices (dofs_per_cell);
  
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Vector displacements (dim); 
  const FEValuesExtractors::Scalar pressure (dim+dim); 
  const FEValuesExtractors::Scalar concentration (dim+dim+1);
 
  std::vector<Vector<double> > old_solution_values (n_q_points, Vector<double>(number_coefficients));
  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (number_coefficients));
  std::vector<Vector<double> > old_solution_face_values (n_face_q_points, Vector<double>(number_coefficients));
  std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (number_coefficients));
  std::vector<Vector<double> > old_timestep_solution_values (n_q_points, Vector<double>(number_coefficients));
  std::vector<std::vector<Tensor<1,dim> > > old_timestep_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (number_coefficients));
  std::vector<Vector<double> > old_timestep_solution_face_values (n_face_q_points, Vector<double>(number_coefficients));    
  std::vector<std::vector<Tensor<1,dim> > > old_timestep_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (number_coefficients));
  std::vector<Vector<double> >   old_timestep_solution_values_inv (n_q_points, Vector<double>(number_coefficients));
   
  typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

        int i = 0;
        int j = 0;
  for (; cell!=endc; ++cell)
  { 
    fe_values.reinit (cell);   
    local_rhs = 0;    
      
    cell_diameter = cell->diameter();
      
    // old Newton iteration
    fe_values.get_function_values (solution, old_solution_values);
    fe_values.get_function_gradients (solution, old_solution_grads);
            
    // old timestep iteration
    fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
    fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);
    
    for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
    {
      fe_values.get_function_values(old_timestep_solution, old_timestep_solution_values); 
    }
    
    // Again, material_id == 0 corresponds to 
    // the domain for fluid equations
    if (cell->material_id() == 0)
    {
      for (unsigned int q=0; q<n_q_points; ++q)
      { 
        int is_on_b = 0;
        double interface_check_x = 0;
        double interface_check_y = 0;
        double co_inv = 0;

        //std::cout << "2" << q << std::endl;
        const double co = ALE_Transformations::get_co<dim> (q, old_solution_values);
        const Tensor<2,dim> pI = ALE_Transformations::get_pI<dim> (q, old_solution_values);
        const Tensor<1,dim> v = ALE_Transformations::get_v<dim> (q, old_solution_values);
        const Tensor<2,dim> grad_v = ALE_Transformations ::get_grad_v<dim> (q, old_solution_grads);
        const Tensor<2,dim> grad_u = ALE_Transformations ::get_grad_u<dim> (q, old_solution_grads);
        const Tensor<1,dim> grad_co = ALE_Transformations ::get_grad_co<dim> (q, old_solution_grads);
        const Tensor<2,dim> grad_v_T = ALE_Transformations::get_grad_v_T<dim> (grad_v);
        const Tensor<1,dim> u = ALE_Transformations::get_u<dim> (q, old_solution_values); 
        const Tensor<2,dim> F = ALE_Transformations::get_F<dim> (q, old_solution_grads);              
        const Tensor<2,dim> F_Inverse = ALE_Transformations::get_F_Inverse<dim> (F);
        const Tensor<2,dim> F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dim> (F_Inverse);
        const double J = ALE_Transformations::get_J<dim> (F);
        
        // This is the fluid stress tensor in ALE formulation
        //TODO anpassen
        const Tensor<2,dim> sigma_ALE = NSE_in_ALE::get_stress_fluid_except_pressure_ALE<dim> (density_fluid, viscosity, grad_v, grad_v_T, F_Inverse, F_Inverse_T );
                              
        // We proceed by catching the previous time step values
        const Tensor<1,dim> old_timestep_v = ALE_Transformations::get_v<dim> (q, old_timestep_solution_values);
        const Tensor<2,dim> old_timestep_grad_v = ALE_Transformations::get_grad_v<dim> (q, old_timestep_solution_grads);
        const Tensor<2,dim> old_timestep_grad_v_T = ALE_Transformations::get_grad_v_T<dim> (old_timestep_grad_v);
        double old_timestep_co = ALE_Transformations::get_co<dim> (q, old_timestep_solution_values);        
        const Tensor<1,dim> old_timestep_u = ALE_Transformations::get_u<dim> (q, old_timestep_solution_values);     
        const Tensor<2,dim> old_timestep_F = ALE_Transformations::get_F<dim> (q, old_timestep_solution_grads);
        const Tensor<2,dim> old_timestep_F_Inverse = ALE_Transformations::get_F_Inverse<dim> (old_timestep_F); 
        const Tensor<2,dim> old_timestep_F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dim> (old_timestep_F_Inverse); 
        const double old_timestep_J = ALE_Transformations::get_J<dim> (old_timestep_F);
               
        // This is the fluid stress tensor in the ALE formulation
        // at the previous time step
        const Tensor<2,dim> old_timestep_sigma_ALE = NSE_in_ALE::get_stress_fluid_except_pressure_ALE<dim> 
                                                    (density_fluid, viscosity, old_timestep_grad_v, old_timestep_grad_v_T, old_timestep_F_Inverse, old_timestep_F_Inverse_T );
        
        Tensor<2,dim> stress_fluid;
        stress_fluid.clear();
        stress_fluid = (J * sigma_ALE * F_Inverse_T);
        
        Tensor<2,dim> fluid_pressure;
        fluid_pressure.clear();
        fluid_pressure = (-pI * J * F_Inverse_T);
                        
        Tensor<2,dim> old_timestep_stress_fluid;
        old_timestep_stress_fluid.clear();
        old_timestep_stress_fluid = (old_timestep_J * old_timestep_sigma_ALE * old_timestep_F_Inverse_T);
    
        // Divergence of the fluid in the ALE formulation
        const double incompressiblity_fluid = NSE_in_ALE::get_Incompressibility_ALE<dim> (q, old_solution_grads);
      
        // Convection term of the fluid in the ALE formulation.
        // We emphasize that the fluid convection term for
        // non-stationary flow problems in ALE
        // representation is difficult to derive.         
        // For adequate discretization, the convection term will be 
        // split into three smaller terms:
        Tensor<1,dim> convection_fluid;
        convection_fluid.clear();
        convection_fluid = density_fluid * J * (grad_v * F_Inverse * v);
             
        // The second convection term for the fluid in the ALE formulation        
        Tensor<1,dim> convection_fluid_with_u;
        convection_fluid_with_u.clear();
        convection_fluid_with_u = density_fluid * J * (grad_v * F_Inverse * u);
        
        // The third convection term for the fluid in the ALE formulation       
        Tensor<1,dim> convection_fluid_with_old_timestep_u;
        convection_fluid_with_old_timestep_u.clear();
        convection_fluid_with_old_timestep_u = density_fluid * J * (grad_v * F_Inverse * old_timestep_u);
        
        // The convection term of the previous time step
        Tensor<1,dim> old_timestep_convection_fluid;
        old_timestep_convection_fluid.clear();
        old_timestep_convection_fluid = (density_fluid * old_timestep_J * (old_timestep_grad_v * old_timestep_F_Inverse * old_timestep_v));

        /******
         *Concentration
         ******/
        // The second convection term for the concentration in the fluid in the ALE formulation   
        double convection_c_fluid = 0;
        for( int l=0; l<dim; l++ )
        {
          for( int m=0; m<dim; m++ )
          {
            convection_c_fluid += J * (grad_co[l] * F_Inverse[l][m] * v[m]);
          }
        }
        double convection_c_fluid_with_u = 0;
        for( int l=0; l<dim; l++ )
        {
          for( int m=0; m<dim; m++ )
          {
            convection_c_fluid_with_u += J * (grad_co[l] * F_Inverse[l][m] * u[m]);
          }
        }
        
        // The third convection term for the concentration in the fluid in the ALE formulation       
        double convection_c_fluid_with_old_timestep_u = 0;
        for( int l=0; l<dim; l++ )
        {
          for( int m=0; m<dim; m++ )
          {
            convection_c_fluid_with_old_timestep_u += J * (grad_co[l] * F_Inverse[l][m] * old_timestep_u[m]);
          }
        }

        /*// The convection term of the previous time step
        double old_timestep_convection_c_fluid;
        //old_timestep_convection_c_fluid.clear();
        old_timestep_convection_c_fluid = ( old_timestep_J * (old_timestep_grad_v * old_timestep_F_Inverse * old_timestep_co));*/

        interface_check_x = 0;
        interface_check_y = 0;
        co_inv = 0; 
      
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          // Fluid, NSE in ALE
          const unsigned int comp_i = fe.system_to_component_index(i).first; 
          if (comp_i == 0 || comp_i == 1)
          {         
            const Tensor<1,dim> phi_i_v = fe_values[velocities].value (i, q);
            const Tensor<2,dim> phi_i_grads_v = fe_values[velocities].gradient (i, q);
          
            local_rhs(i) -= (compute_short_scale * density_fluid * (J + old_timestep_J)/2.0 * 
                (v - old_timestep_v) * phi_i_v +       
                timestep * theta * convection_fluid * phi_i_v +  
                timestep * (1.0-theta) *
                old_timestep_convection_fluid * phi_i_v -
                (convection_fluid_with_u -
                convection_fluid_with_old_timestep_u) * phi_i_v +
                timestep * scalar_product(fluid_pressure, phi_i_grads_v) +
                timestep * theta * scalar_product(stress_fluid, phi_i_grads_v) +
                timestep * (1.0-theta) *
                scalar_product(old_timestep_stress_fluid, phi_i_grads_v)       
                ) *  fe_values.JxW(q);
          }   
          else if (comp_i == 2 || comp_i == 3)
          { 
            const Tensor<2,dim> phi_i_grads_u = fe_values[displacements].gradient (i, q);

            local_rhs(i) -= ( alpha_u/J * scalar_product(grad_u, phi_i_grads_u)
                              ) * fe_values.JxW(q);
          }  
          else if (comp_i == 4)
          {
            const double phi_i_p = fe_values[pressure].value (i, q);
            local_rhs(i) -= (incompressiblity_fluid * phi_i_p) *  fe_values.JxW(q);
          } 
          else if (comp_i == 5)
          {
            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell->at_boundary(face))
                continue;
              

              if (cell->face(face)->has_children())
              {
                for (unsigned int subface = 0; subface < cell->face(face)->n_children(); ++subface)
                {
                    const auto neighbor_child = cell->neighbor_child_on_subface(face, subface);

                    // Jetzt: cell und neighbor_child bilden ein Interface

                    // Voraussetzung: Materialunterschied (fluid ↔ solid)
                    if ( cell->material_id() != neighbor_child->material_id() )
                    {
                      is_on_b = 1.;
                      continue;
                    }

                    fe_face_values_neighbor.reinit(neighbor_child, neighbor_child->neighbor_of_neighbor(face), 0);

                    for (unsigned int q = 0; q < fe_face_values_neighbor.n_quadrature_points; ++q)
                    {  
                      co_inv = fe_face_values_neighbor.shape_value(i, q); 
                    }
                }
              }
            }

            const double phi_i_c = fe_values[concentration].value (i, q);
            const Tensor<1,dim> phi_i_grads_c = fe_values[concentration].gradient (i, q);
            local_rhs(i) -= ( compute_short_scale * (J + old_timestep_J)/2.0 * 
                              (co - old_timestep_co) * phi_i_c
                              + timestep * theta * J * Df * grad_co * F_Inverse * F_Inverse_T * phi_i_grads_c
                              + timestep * theta * convection_c_fluid * phi_i_c
                              - convection_c_fluid_with_u * phi_i_c
                              + convection_c_fluid_with_old_timestep_u * phi_i_c
                              + timestep * theta * is_on_b * kd * co_inv * phi_i_c
                            ) * fe_values.JxW(q);
          }
        } // end i dofs   
      } // close n_q_points  

      // Pressure Neumann conditions on inflow boundary
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() == 0) )
        {
          fe_face_values.reinit (cell, face);        
       
          for (unsigned int q=0; q<n_face_q_points; ++q)
          { 
            Tensor<1,dim> neumann_value;
            neumann_value[0] = neumann_pressure_x;
            neumann_value[1] = neumann_pressure_y;
          
            //double fluid_pressure = old_timestep_solution_face_values[q](dim+dim);

            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const unsigned int comp_i = fe.system_to_component_index(i).first; 
              if (comp_i == 0 || comp_i == 1)
              {  
                local_rhs(i) +=  (timestep * neumann_value * fe_face_values[velocities].value (i, q)       
                    ) * fe_face_values.JxW(q);             
              }
            // end i
            }  
          // end face_n_q_points    
          }                                     
        } 
      }  // end face integrals
                
      // As already discussed in the assembling method for the matrix,
      // we have to integrate some terms on the outflow boundary:
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() &&      
              (cell->face(face)->boundary_id() == 1))
        {
          fe_face_values.reinit (cell, face);
      
          fe_face_values.get_function_values (solution, old_solution_face_values);
          fe_face_values.get_function_gradients (solution, old_solution_face_grads);
      
          fe_face_values.get_function_values (old_timestep_solution, old_timestep_solution_face_values);
          fe_face_values.get_function_gradients (old_timestep_solution, old_timestep_solution_face_grads);      
      
          for (unsigned int q=0; q<n_face_q_points; ++q)
          { 
            // These are terms coming from the
            // previous Newton iterations ...
            const Tensor<2,dim> grad_v = ALE_Transformations::get_grad_v<dim> (q, old_solution_face_grads);
            const Tensor<2,dim> grad_v_T = ALE_Transformations::get_grad_v_T<dim> (grad_v);
            const Tensor<1,dim> grad_co = ALE_Transformations::get_grad_co<dim> (q, old_solution_grads); 
            const Tensor<2,dim> F = ALE_Transformations::get_F<dim> (q, old_solution_face_grads);
            const Tensor<2,dim> F_Inverse = ALE_Transformations::get_F_Inverse<dim> (F);
            const Tensor<2,dim> F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dim> (F_Inverse);
            const double J = ALE_Transformations::get_J<dim> (F);
          
            // ... and here from the previous time step iteration
            const Tensor<2,dim> old_timestep_grad_v = ALE_Transformations::get_grad_v<dim> (q, old_timestep_solution_face_grads);
            const Tensor<2,dim> old_timestep_grad_v_T = ALE_Transformations::get_grad_v_T<dim> (old_timestep_grad_v);
            const Tensor<2,dim> old_timestep_F = ALE_Transformations::get_F<dim> (q, old_timestep_solution_face_grads);
            const Tensor<2,dim> old_timestep_F_Inverse = ALE_Transformations::get_F_Inverse<dim> (old_timestep_F);
            const Tensor<2,dim> old_timestep_F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dim> (old_timestep_F_Inverse);
            const double old_timestep_J = ALE_Transformations::get_J<dim> (old_timestep_F);
                
            Tensor<2,dim> sigma_ALE_tilde;
            sigma_ALE_tilde.clear();
            sigma_ALE_tilde = (density_fluid * viscosity * F_Inverse_T * grad_v_T);
          
            Tensor<2,dim> old_timestep_sigma_ALE_tilde;
            old_timestep_sigma_ALE_tilde.clear();
            old_timestep_sigma_ALE_tilde = (density_fluid * viscosity * old_timestep_F_Inverse_T * old_timestep_grad_v_T);
          
            // Neumann boundary integral
            Tensor<2,dim> stress_fluid_transposed_part;
            stress_fluid_transposed_part.clear();
            stress_fluid_transposed_part = (J * sigma_ALE_tilde * F_Inverse_T);
            
            Tensor<2,dim> old_timestep_stress_fluid_transposed_part;
            old_timestep_stress_fluid_transposed_part.clear();          
            old_timestep_stress_fluid_transposed_part = (old_timestep_J * old_timestep_sigma_ALE_tilde * old_timestep_F_Inverse_T);

            const Tensor<1,dim> neumann_value = (stress_fluid_transposed_part * fe_face_values.normal_vector(q));
            const Tensor<1,dim> old_timestep_neumann_value = (old_timestep_stress_fluid_transposed_part * fe_face_values.normal_vector(q));
                   
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const unsigned int comp_i = fe.system_to_component_index(i).first; 
              if (comp_i == 0 || comp_i == 1)
              {  
                local_rhs(i) +=  (timestep * theta * 
                                 neumann_value * fe_face_values[velocities].value (i, q) +
                                 timestep * (1.0-theta) *
                                 old_timestep_neumann_value * 
                                 fe_face_values[velocities].value (i, q)
                                 ) * fe_face_values.JxW(q);
              }
            } // end i  
          } // end face_n_q_points                 
        } 
      }  // end face integrals do-nothing condition

      // As already discussed in the assembling method for the matrix,
      // we have to integrate some terms on the outflow boundary:
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() == 0) )
        {
          fe_face_values.reinit (cell, face);
      
          fe_face_values.get_function_values (solution, old_solution_face_values);
          fe_face_values.get_function_gradients (solution, old_solution_face_grads);
      
          fe_face_values.get_function_values (old_timestep_solution, old_timestep_solution_face_values);
          fe_face_values.get_function_gradients (old_timestep_solution, old_timestep_solution_face_grads);      
      
          for (unsigned int q=0; q<n_face_q_points; ++q)
          { 
            // These are terms coming from the
            // previous Newton iterations ...
            const Tensor<2,dim> grad_v = ALE_Transformations::get_grad_v<dim> (q, old_solution_face_grads);
            const Tensor<2,dim> grad_v_T = ALE_Transformations::get_grad_v_T<dim> (grad_v);
            const Tensor<2,dim> F = ALE_Transformations::get_F<dim> (q, old_solution_face_grads);
            const Tensor<2,dim> F_Inverse = ALE_Transformations::get_F_Inverse<dim> (F);
            const Tensor<2,dim> F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dim> (F_Inverse);
            const double J = ALE_Transformations::get_J<dim> (F);
          
            // ... and here from the previous time step iteration
            const Tensor<2,dim> old_timestep_grad_v = ALE_Transformations::get_grad_v<dim> (q, old_timestep_solution_face_grads);
            const Tensor<2,dim> old_timestep_grad_v_T = ALE_Transformations::get_grad_v_T<dim> (old_timestep_grad_v);
            const Tensor<2,dim> old_timestep_F = ALE_Transformations::get_F<dim> (q, old_timestep_solution_face_grads);
            const Tensor<2,dim> old_timestep_F_Inverse = ALE_Transformations::get_F_Inverse<dim> (old_timestep_F);
            const Tensor<2,dim> old_timestep_F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dim> (old_timestep_F_Inverse);
            const double old_timestep_J = ALE_Transformations::get_J<dim> (old_timestep_F);
                
            Tensor<2,dim> sigma_ALE_tilde;
            sigma_ALE_tilde.clear();
            sigma_ALE_tilde = (density_fluid * viscosity * F_Inverse_T * grad_v_T);
          
            Tensor<2,dim> old_timestep_sigma_ALE_tilde;
            old_timestep_sigma_ALE_tilde.clear();
            old_timestep_sigma_ALE_tilde = (density_fluid * viscosity * old_timestep_F_Inverse_T * old_timestep_grad_v_T);
          
            // Neumann boundary integral
            Tensor<2,dim> stress_fluid_transposed_part;
            stress_fluid_transposed_part.clear();
            stress_fluid_transposed_part = (J * sigma_ALE_tilde * F_Inverse_T);
          
            Tensor<2,dim> old_timestep_stress_fluid_transposed_part;
            old_timestep_stress_fluid_transposed_part.clear();          
            old_timestep_stress_fluid_transposed_part = (old_timestep_J * old_timestep_sigma_ALE_tilde * old_timestep_F_Inverse_T);

            // no singularities at t=0 at inflow
            Tensor<2,dim> pI_inflow;
            pI_inflow[0][0] = -0.0 * (1.0 - 1.0*std::cos(0.1*pi*time)); //-pressure_fluid_x;
            pI_inflow[1][1] = -0.0 * (1.0 - 1.0*std::cos(0.1*pi*time));//-pressure_fluid_x;

            const Tensor<1,dim> neumann_value_stress = (stress_fluid_transposed_part * fe_face_values.normal_vector(q));
            const Tensor<1,dim> neumann_value = (pI_inflow * fe_face_values.normal_vector(q));

            //Concentration
            const double co = ALE_Transformations::get_co<dim> (q, old_solution_values);
            const Tensor<1,dim> v = ALE_Transformations::get_v<dim> (q, old_solution_values);
            const Tensor<1,dim> grad_co = ALE_Transformations::get_grad_co<dim> (q, old_solution_grads); 
                   
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const unsigned int comp_i = fe.system_to_component_index(i).first; 
              if (comp_i == 0 || comp_i == 1)
              {  
                local_rhs(i) +=  (timestep * theta * 
                                (neumann_value + neumann_value_stress) * fe_face_values[velocities].value (i, q) 
                                ) * fe_face_values.JxW(q);            
              }
            } // end i
          } // end face_n_q_points                                      
        } 
      }  // end face integrals do-nothing condition inflow

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (local_rhs, local_dof_indices,
                system_rhs);
   
      // Finally, we arrive at the end for assembling 
      // the variational formulation for the fluid part and step to
      // the assembling process of the structure terms:
    }   
    else if (cell->material_id() == 1)
    {
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        int is_on_b = 0;
        double interface_check_x = 0;
        double interface_check_y = 0;
        double co_inv = 0; //adhesion/attachement

        const double co = ALE_Transformations::get_co<dim> (q, old_solution_values);
        const Tensor<1,dim> v = ALE_Transformations::get_v<dim> (q, old_solution_values);
        const Tensor<1,dim> grad_co = ALE_Transformations::get_grad_co<dim> (q, old_solution_grads); 
        const Tensor<1,dim> u = ALE_Transformations::get_u<dim> (q, old_solution_values); 
        const Tensor<2,dim> F = ALE_Transformations::get_F<dim> (q, old_solution_grads);
        const Tensor<2,dim> F_T = ALE_Transformations::get_F_T<dim> (F);
        const Tensor<2,dim> Identity = ALE_Transformations::get_Identity<dim> ();
        const Tensor<2,dim> F_Inverse = ALE_Transformations::get_F_Inverse<dim> (F);
        const Tensor<2,dim> F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dim> (F_Inverse);
        const double J = ALE_Transformations::get_J<dim> (F);
        
        // Previous time step values
        const Tensor<1,dim> old_timestep_v = ALE_Transformations::get_v<dim> (q, old_timestep_solution_values);
        const Tensor<1,dim> old_timestep_u = ALE_Transformations::get_u<dim> (q, old_timestep_solution_values);
        double old_timestep_co = ALE_Transformations::get_co<dim> (q, old_timestep_solution_values);
        const Tensor<2,dim> old_timestep_F = ALE_Transformations::get_F<dim> (q, old_timestep_solution_grads);
        const Tensor<2,dim> old_timestep_F_Inverse = ALE_Transformations::get_F_Inverse<dim> (old_timestep_F);
        const Tensor<2,dim> old_timestep_F_T = ALE_Transformations::get_F_T<dim> (old_timestep_F);
        const Tensor<2,dim> old_timestep_F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dim> (old_timestep_F_Inverse);
        const double old_timestep_J = ALE_Transformations::get_J<dim> (old_timestep_F);
        
        // Growth
        double g_growth = 0.0;
        if ((std::abs(fe_values.quadrature_point(q)[1]) > 1.0) && fe_values.quadrature_point(q)[1] > 0 && (time <=stop_growth))
        {
          g_growth = 1.0 + 1e-02 * old_timestep_co * 
                    std::exp(-(fe_values.quadrature_point(q)[0]-growth_displacement) * (fe_values.quadrature_point(q)[0]-growth_displacement))
                    * (2.0 - std::abs(fe_values.quadrature_point(q)[1]));
        }
        else if ((std::abs(fe_values.quadrature_point(q)[1]) > 1.0) && fe_values.quadrature_point(q)[1] < 0 && (time <=stop_growth))
        {
          g_growth = 1.0 + 1e-02 * old_timestep_co * 
                    std::exp(-(fe_values.quadrature_point(q)[0]+growth_displacement) * (fe_values.quadrature_point(q)[0]+growth_displacement))
                    * (2.0 - std::abs(fe_values.quadrature_point(q)[1]));
        }
        else if ((std::abs(fe_values.quadrature_point(q)[1]) > 1.0) && (time > stop_growth))
        {
          abort();
        }
        else 
          g_growth = 1.0;

        double g_c_growth = 0;
        if( fe_values.quadrature_point(q)[1] > 0 )
        {
          if ((std::abs(fe_values.quadrature_point(q)[0]-growth_displacement) < 2) )
          {
            g_c_growth = 1.;
          }
          else if( (std::abs(fe_values.quadrature_point(q)[0]-growth_displacement) < 2.5) )
          {
            g_c_growth = 1-2*(std::abs(fe_values.quadrature_point(q)[0]-growth_displacement) -2 );
          }
          else
          {
            g_c_growth = 0;
          }
        }
        else if( fe_values.quadrature_point(q)[1] < 0 )
        {
          if ((std::abs(fe_values.quadrature_point(q)[0]+growth_displacement) < 2) )
          {
            g_c_growth = 1.;
          }
          else if( (std::abs(fe_values.quadrature_point(q)[0]+growth_displacement) < 2.5) )
          {
            g_c_growth = 1-2*(std::abs(fe_values.quadrature_point(q)[0]+growth_displacement) -2 );
          }
          else
          {
            g_c_growth = 0;
          }
        }

        if(g_c_growth > 1.)
          std::cout << "WRONG!!!!" << std::endl;

        const Tensor<2,dim> E = Structure_Terms_in_ALE::get_E<dim> (F_T, F, Identity, g_growth);
        const double tr_E = Structure_Terms_in_ALE::get_tr_E<dim> (E);

        const Tensor<2,dim> old_timestep_E = Structure_Terms_in_ALE::get_E<dim> (old_timestep_F_T, old_timestep_F, Identity, g_growth);
        const double old_timestep_tr_E = Structure_Terms_in_ALE::get_tr_E<dim> (old_timestep_E);

        // STVK structure model
        Tensor<2,dim> sigma_structure_ALE;
        sigma_structure_ALE.clear();
        sigma_structure_ALE = lame_coefficient_lambda * tr_E * Identity 
                              + 2 * lame_coefficient_mu * E ;
        
        Tensor<2,dim> stress_term;
        stress_term.clear();
        stress_term = sigma_structure_ALE;
        
        Tensor<2,dim> old_timestep_sigma_structure_ALE;
        old_timestep_sigma_structure_ALE.clear();
        old_timestep_sigma_structure_ALE = (1.0/old_timestep_J *
              old_timestep_F * (lame_coefficient_lambda * 1.0/g_growth *
                    old_timestep_tr_E * Identity +
                    2 * lame_coefficient_mu * 1.0/g_growth *
                    old_timestep_E) * 
              old_timestep_F_T);

        is_on_b = 0;
        interface_check_x = 0;
        interface_check_y = 0;
        
        Tensor<2,dim> old_timestep_stress_term;
        old_timestep_stress_term.clear();
        old_timestep_stress_term = (old_timestep_J * old_timestep_sigma_structure_ALE * old_timestep_F_Inverse_T);
                
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          // STVK structure model
          const unsigned int comp_i = fe.system_to_component_index(i).first; 
          if (comp_i == 0 || comp_i == 1)
          { 
            const Tensor<1,dim> phi_i_v = fe_values[velocities].value (i, q);
            const Tensor<2,dim> phi_i_grads_v = fe_values[velocities].gradient (i, q);
          
            local_rhs(i) -= (density_structure * (v - old_timestep_v) * phi_i_v 
                           + timestep * theta * scalar_product(stress_term,phi_i_grads_v)
                           ) * fe_values.JxW(q);
          }   
          else if (comp_i == 2 || comp_i == 3)
          {
            const Tensor<1,dim> phi_i_u =  fe_values[displacements].value (i, q);
            local_rhs(i) -=  (density_structure * alpha_us * 
            ((u - old_timestep_u) * phi_i_u 
              -timestep * (theta * v + (1.0-theta) * old_timestep_v) * phi_i_u)
            ) * fe_values.JxW(q);   
          }
          else if (comp_i == 4)
          {
            const double phi_i_p = fe_values[pressure].value (i, q);
            local_rhs(i) -= (old_solution_values[q](dim+dim) * phi_i_p) * fe_values.JxW(q);   
          }
          else if (comp_i == 5)
          {
            const double phi_i_c = fe_values[concentration].value (i, q);
            const Tensor<1,dim> phi_i_grads_c = fe_values[concentration].gradient (i, q);

            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell->at_boundary(face))
                continue;

              if (cell->face(face)->has_children())
              {
                for (unsigned int subface = 0; subface < cell->face(face)->n_children(); ++subface)
                {
                    const auto neighbor_child = cell->neighbor_child_on_subface(face, subface);

                    // Jetzt: cell und neighbor_child bilden ein Interface

                    // Voraussetzung: Materialunterschied (fluid ↔ solid)
                    if ( cell->material_id() != neighbor_child->material_id() )
                    {
                      is_on_b = 1.;
                      continue;
                    }

                    fe_face_values_neighbor.reinit(neighbor_child, neighbor_child->neighbor_of_neighbor(face), 0);

                    for (unsigned int q = 0; q < fe_face_values_neighbor.n_quadrature_points; ++q)
                    {  
                      co_inv = fe_face_values_neighbor.shape_value(i, q); 
                    }
                }
              }
            }
          
            local_rhs(i) -= ( compute_short_scale * (co - old_timestep_co) * phi_i_c
                            - timestep * theta * grad_co * v * phi_i_c 
                            - timestep * theta * co * v * phi_i_grads_c
                            + timestep * theta * Ds * grad_co * phi_i_grads_c 
                            - timestep * c_growth * old_timestep_co * phi_i_c * g_c_growth 
                            + timestep * theta * is_on_b * ka * co_inv * phi_i_c

                            ) *  fe_values.JxW(q);
          }
        } // end i    
      } // end n_q_points 
      
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (local_rhs, local_dof_indices,
                system_rhs);
      
    
    } // end if (for STVK material)  
  }  // end cell   
  timer.leave_subsection(); 
}

template <int dim>
void
FSI_ALE_Problem<dim>::set_initial_condition( )
{
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  double c_initial = 0;
  const unsigned int block = 3; 

  for (unsigned int i = 0; i < solution.block(block).size(); ++i)//, ++cell)
  {
    const unsigned int lokaler_index = i;  // Index in block

    // Block-Offset
    unsigned int offset = 0;
    for (unsigned int b = 0; b < block; ++b)
    {
        offset += solution.block(b).size(); 
    }

    const unsigned int globaler_index = offset + lokaler_index;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        std::vector<types::global_dof_index> dof_indices(cell->get_fe().dofs_per_cell);
        cell->get_dof_indices(dof_indices);

        if (std::find(dof_indices.begin(), dof_indices.end(), globaler_index) != dof_indices.end())
        {
            c_initial = (cell->material_id() == 1) ? 10.0 : 2.0;
            break;
        }
    }
    //c_initial = (cell->material_id() == 1) ? 10.0 : 2.0;
    solution.block(block)(i) = c_initial;
  }
}


// Here, we impose boundary conditions
// for the whole system. The fluid inflow 
// is prescribed by a parabolic profile. The usual
// structure displacement shall be fixed  
// at all outer boundaries. Consequently
// our formulation of the mixed biharmonic equation
// requires no Dirichlet zero values for the 
// second displacement variable $w$ (see the
// standard literature to elasticity and Ciarlet).
// The pressure variable is not subjected to any
// Dirichlet boundary conditions and is left free 
// in this method. Please note, that 
// the interface between fluid and structure has no
// physical boundary due to our formulation. Interface
// conditions are automatically fulfilled: that is 
// one major advantage of the `monolithic' formulation.
template <int dim>
void
FSI_ALE_Problem<dim>::set_initial_bc(const double time)
{
  std::map<types::global_dof_index, double> boundary_values;  
  ComponentMask displacement_mask(number_coefficients, false);
  displacement_mask.set(0, true);             // ux
  displacement_mask.set(1, true);              // uy
  displacement_mask.set(dim + dim + 1, true);  // evtl. w-Komponente

  // Maske für feste Struktur (alle Displacements fixieren, Druck bleibt frei)
  ComponentMask structure_mask(number_coefficients, false);
  structure_mask.set(dim + dim, true); // Druck bleibt frei

  // Parabolisches Inflow-Profil auf Rand 0 und 1
  VectorTools::interpolate_boundary_values(
    dof_handler,
    0,
    BoundaryParabel<dim>(time, u_y, compute_short_scale),
    boundary_values,
    displacement_mask);

  VectorTools::interpolate_boundary_values(
    dof_handler,
    1,
    BoundaryParabel<dim>(time, u_y, compute_short_scale),
    boundary_values,
    displacement_mask);

  // Feste Struktur auf Rändern 2, 3, 80, 82
  for (const auto boundary_id : {2, 3, 80, 82})
  {
    VectorTools::interpolate_boundary_values(
      dof_handler,
      boundary_id,
      Functions::ZeroFunction<dim>(number_coefficients),
      boundary_values,
      structure_mask);
  }

  int j = 0;
  for (typename std::map<types::global_dof_index, double>::const_iterator
     i = boundary_values.begin();
   i != boundary_values.end();
   ++i)
      solution(i->first) = i->second;
}

// This function applies boundary conditions 
// to the Newton iteration steps. For all variables that
// have Dirichlet conditions on some (or all) parts
// of the outer boundary, we apply zero-Dirichlet
// conditions, now. 
template <int dim>
void
FSI_ALE_Problem<dim>::set_newton_bc ()
{
  std::vector<bool> component_mask (number_coefficients, true);
  component_mask[dim+dim] = false; 

  component_mask[dim+dim+1] = true; //false; 

  // Because of Pressure inflow
  component_mask[0] = true;
  component_mask[1] = true;      
  VectorTools::interpolate_boundary_values (dof_handler,
                0,
                dealii::Functions::ZeroFunction<dim>(number_coefficients),                                             
                constraints,
                component_mask); 
   
  component_mask[0] = true;
  component_mask[1] = true;  
  component_mask[dim+dim+1] = false;     

  VectorTools::interpolate_boundary_values (dof_handler,
                                              2,
                dealii::Functions::ZeroFunction<dim>(number_coefficients),  
                                              constraints,
                                              component_mask);

  VectorTools::interpolate_boundary_values (dof_handler,
                                              7,
                dealii::Functions::ZeroFunction<dim>(number_coefficients),  
                                              constraints,
                                              component_mask);


  VectorTools::interpolate_boundary_values (dof_handler,
                                              3,
                dealii::Functions::ZeroFunction<dim>(number_coefficients),  
                                              constraints,
                                              component_mask);

  VectorTools::interpolate_boundary_values (dof_handler,
                                              80,
                dealii::Functions::ZeroFunction<dim>(number_coefficients),  
                                              constraints,
                                              component_mask);
  VectorTools::interpolate_boundary_values (dof_handler,
                82,
                dealii::Functions::ZeroFunction<dim>(number_coefficients),  
                constraints,
                component_mask);   

  component_mask[0] = false;
  component_mask[1] = false;
  component_mask[dim+dim+1] = false;    
    
  VectorTools::interpolate_boundary_values (dof_handler,
                1,
                dealii::Functions::ZeroFunction<dim>(number_coefficients),  
                constraints,
                component_mask);
}  

// In this function, we solve the linear systems
// inside the nonlinear Newton iteration. We only
// use a direct solver from UMFPACK.
// The reason is twofold:
// First, the focus of this implementation is 
// more on time-dependent problems. Hence, a huge 
// amount of spatial degrees of freedom is not our 
// primal goal. For this, a direct solver is an adequate tool.
// Second, the devolpement of an iterative solver based on the
// GMRES scheme for instance, is difficult to derive. Only a few
// results are known in the literature so far. However,
// Baerbel Janssen and the author have already had a first try 
// for our implementation at hand (that is working, of coarse) but suffers
// from a good preconditioner for the GMRES scheme 
// (Lit. B. Janssen, T. Wick, ECCOMAS 2010). This lack
// will be resolved in upcoming work but we invite everybody to 
// collaborate with us if he/she has a resonable idea.   
template <int dim>
void 
FSI_ALE_Problem<dim>::solve () 
{
  timer.enter_subsection("Solve linear system.");
  Vector<double> sol, rhs;    
  sol = newton_update;    
  rhs = system_rhs;
    
  A_direct.vmult(sol,rhs); 
  newton_update = sol;
  
  constraints.distribute (newton_update);
  timer.leave_subsection(); 
}

template <int dim>
void FSI_ALE_Problem<dim>::refine_grid()
{
  // Fehlerindikator berechnen (Konzentration)
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(
    dof_handler,
    QGauss<dim - 1>(degree + 1),
    {},
    solution,
    estimated_error_per_cell,
    fe.component_mask(FEValuesExtractors::Scalar(5))
  );

  // Schritt 4: Refinement-Markierung
  GridRefinement::refine_and_coarsen_fixed_number(
    triangulation, estimated_error_per_cell,
    0.3, 0.03);

  // Refinement-Level begrenzen
  const unsigned int max_level = 3;
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->level() >= max_level)
      cell->clear_refine_flag();

  BlockVector<double> tmp_solution;
  tmp_solution = solution;

  SolutionTransfer<dim, BlockVector<double> > solution_transfer (dof_handler);

  triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(tmp_solution);

  double n_active_before = triangulation.n_active_cells();

  triangulation.execute_coarsening_and_refinement();

  double n_active_after = triangulation.n_active_cells();
  std::cout << std::setprecision(5) << "Zellen vorher: " << n_active_before
            << ", nachher: " << n_active_after
            << ", netto hinzugefügt: " << (n_active_after - n_active_before)
            << std::endl;
  setup_timestep ();

  solution_transfer.interpolate(tmp_solution, solution);
}

// This is the Newton iteration to solve the 
// non-linear system of equations. First, we declare some
// standard parameters of the solution method. Addionally,
// we also implement an easy line search algorithm. 
template <int dim>
void FSI_ALE_Problem<dim>::newton_iteration (const double time) 
                 
{ 
  Timer timer_newton;
  const double lower_bound_newton_residuum = 1.0e-8;

  // TODO
  const unsigned int max_no_newton_steps  = 100;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 0.1; 
 
  // Line search parameters
  unsigned int line_search_step;
  const unsigned int  max_no_line_search_steps = 10;
  const double line_search_damping = 0.6;
  double new_newton_residuum;
  unsigned int stop_when_line_search_two_times_max_number = 0;
  
  // Application of the initial boundary conditions to the 
  // variational equations:
  set_initial_bc (time);

  assemble_system_rhs();

  double newton_residuum = system_rhs.linfty_norm(); 
  double old_newton_residuum= newton_residuum;
  double initial_newton_residuum = newton_residuum;
  unsigned int newton_step = 1;
   
  if (newton_residuum < lower_bound_newton_residuum)
  {
    std::cout << '\t' << std::scientific << newton_residuum << std::endl;     
  }
  
  while ((newton_residuum > lower_bound_newton_residuum &&
    (newton_residuum/initial_newton_residuum) > lower_bound_newton_residuum) &&
   newton_step < max_no_newton_steps )
  {
    timer_newton.start();
    old_newton_residuum = newton_residuum;
      
    assemble_system_rhs();
    newton_residuum = system_rhs.linfty_norm();

    if (newton_residuum < lower_bound_newton_residuum)
    {
      std::cout << '\t' << std::scientific << newton_residuum << std::endl;
      break;
    }
  
    if (newton_residuum/old_newton_residuum > nonlinear_rho)
    {
      assemble_system_matrix ();
      std::set<unsigned int> null_dofs;

      // Only factorize when matrix is re-built
      A_direct.factorize(system_matrix);   
    }

    // Solve Ax = b
    solve ();   
        
    line_search_step = 0;   
    for ( ; line_search_step < max_no_line_search_steps; ++line_search_step)
    {                
      solution += newton_update;
      constraints.distribute(solution);
    
      assemble_system_rhs ();     
      new_newton_residuum = system_rhs.linfty_norm();
    
      if (new_newton_residuum < newton_residuum)
        break;
      else    
        solution -= newton_update;
    
      newton_update *= line_search_damping;
    }    
     
    timer_newton.stop();
    if (line_search_step == 10)
      stop_when_line_search_two_times_max_number ++;

    if (stop_when_line_search_two_times_max_number == 3)
    {
      std::cout << "Aborting Newton as line search does not help to converge anymore." << std::endl;
      abort();
    }
      
    std::cout << std::setprecision(5) <<newton_step << '\t' 
              << std::scientific << newton_residuum << '\t'
              << std::scientific << newton_residuum/old_newton_residuum  <<'\t' ;
    if (newton_residuum/old_newton_residuum > nonlinear_rho)
      std::cout << "r" << '\t' ;
    else 
      std::cout << " " << '\t' ;
    std::cout << line_search_step  << '\t' << std::scientific << timer_newton.cpu_time () << std::endl;

    // Updates
    timer_newton.reset();
    newton_step++;      
  }
}

// This function is known from almost all other 
// tutorial steps in deal.II.
template <int dim>
void
FSI_ALE_Problem<dim>::output_results (const unsigned int refinement_cycle,
            const BlockVector<double> output_vector)  const
{
  std::vector<std::string> solution_names; 
  solution_names.push_back ("x_velo");
  solution_names.push_back ("y_velo"); 
  solution_names.push_back ("x_dis");
  solution_names.push_back ("y_dis");
  solution_names.push_back ("p_fluid");
  solution_names.push_back ("conc");
   
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim+dim+1+1, DataComponentInterpretation::component_is_scalar);


  
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);  
   
  data_out.add_data_vector (output_vector, solution_names,
          DataOut<dim>::type_dof_data,
          data_component_interpretation);
  
  data_out.build_patches ();

  std::string filename_basis;
  filename_basis  = "solution_fsi_case_1_beta_01_global_2_"; 
   
  std::ostringstream filename;

  std::cout << "------------------" << std::endl;
  std::cout << "Write solution" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;
  filename << filename_basis
     << Utilities::int_to_string (refinement_cycle, 5)
     << ".vtk";
  
  std::ofstream output (filename.str().c_str());
  data_out.write_vtk (output);

}

template <int dim>
void FSI_ALE_Problem<dim>::compute_minimal_J()
{
  QGauss<dim>   quadrature_formula(degree+2);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);
  const unsigned int   n_q_points      = quadrature_formula.size();
  
  
  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, 
                std::vector<Tensor<1,dim> > (number_coefficients));
  
  double min_J= 1.0e+5;
  double J=1.0e+5;
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  
  for (; cell!=endc; ++cell)
  { 
    fe_values.reinit (cell);
      
    // old Newton iteration 
    fe_values.get_function_gradients (solution, old_solution_grads);

    for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Tensor<2,dim> F = ALE_Transformations::get_F<dim> (q, old_solution_grads);
      
      J = ALE_Transformations::get_J<dim> (F);
      if (J < min_J)
        min_J = J;
    }
  }
  
  if (compute_short_scale)    
    std::cout << "SScMin J: " << timestep_number << "   " << time << "  " << min_J << std::endl;
  else 
    std::cout << "LScMin J: " << timestep_number << "   " << time << "  " << min_J << std::endl;
}

template <int dim>
void FSI_ALE_Problem<dim>::compute_line_velocity(double eval_line)
{
  const QGauss<dim-1> face_quadrature_formula (degree+2);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
            update_values | update_quadrature_points | 
            update_gradients | update_normal_vectors | 
            update_JxW_values);
  
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<Vector<double> >  face_solution_values (n_face_q_points, 
                  Vector<double> (number_coefficients));

  std::vector<std::vector<Tensor<1,dim> > > 
    face_solution_grads (n_face_q_points, std::vector<Tensor<1,dim> > (number_coefficients));
  
  
  double value = 0.0;
  
  double eps = 1.0e-6;

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  for (; cell!=endc; ++cell)
  {
    for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
    {
      fe_face_values.reinit (cell, face);
      fe_face_values.get_function_values (solution, face_solution_values);
      fe_face_values.get_function_gradients (solution, face_solution_grads);
     
      for (unsigned int q=0; q<n_face_q_points; ++q)
      {
        if ((fe_face_values.quadrature_point(q)[1] < (eval_line + eps))
            && (fe_face_values.quadrature_point(q)[1] > (eval_line - eps)))
        {
          double u;
          u = face_solution_values[q](0); 
       
          value += u * fe_face_values.JxW(q); 
        } // end middle line id
      } // end face q points 
    } // end faces
  } // end element
   std::cout << "Middle line u:   "  << "   " << std::setprecision(16) << value << std::endl;
}

// With help of this function, we extract 
// point values for a certain component from our
// discrete solution. We use it to gain the 
// displacements of the structure in the x- and y-directions.
template <int dim>
double FSI_ALE_Problem<dim>::compute_point_value (Point<dim> p, 
                 const unsigned int component) const  
{
 
  Vector<double> tmp_vector(number_coefficients);
  VectorTools::point_value (dof_handler, 
          solution, 
          p, 
          tmp_vector);
  
  return tmp_vector(component);
}


template <int dim>
void FSI_ALE_Problem<dim>::compute_drag_lift_fsi_fluid_tensor()
{
  const QGauss<dim-1> face_quadrature_formula (3);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
            update_values | update_gradients | update_normal_vectors | 
            update_JxW_values);
  
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<long unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<Vector<double> >  face_solution_values (n_face_q_points, 
                  Vector<double> (number_coefficients));

  std::vector<std::vector<Tensor<1,dim> > > 
    face_solution_grads (n_face_q_points, std::vector<Tensor<1,dim> > (number_coefficients));
  
  Tensor<1,dim> drag_lift_value;
  
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell!=endc; ++cell)
  {
    // Now, we compute the forces that act on the beam. Here,
    // we have two possibilities as already discussed in the paper.
    // We use again the fluid tensor to compute 
    // drag and lift:
    if (cell->material_id() == 0)
    {     
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        if (cell->neighbor_index(face) != -1)         
          if (cell->material_id() !=  cell->neighbor(face)->material_id())
          {
            fe_face_values.reinit (cell, face);
            fe_face_values.get_function_values (solution, face_solution_values);
            fe_face_values.get_function_gradients (solution, face_solution_grads);
            
            for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
            {
              const Tensor<2,dim> pI = ALE_Transformations::get_pI<dim> (q_point, face_solution_values);
              const Tensor<2,dim> grad_v = ALE_Transformations ::get_grad_v<dim> (q_point, face_solution_grads);
              const Tensor<1,dim> grad_co = ALE_Transformations::get_grad_co<dim> (q_point, face_solution_grads);
              const Tensor<2,dim> grad_v_T = ALE_Transformations::get_grad_v_T<dim> (grad_v);
              const Tensor<2,dim> F = ALE_Transformations::get_F<dim> (q_point, face_solution_grads);              
              const Tensor<2,dim> F_Inverse = ALE_Transformations::get_F_Inverse<dim> (F);
              const Tensor<2,dim> F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dim> (F_Inverse);
              const double J = ALE_Transformations::get_J<dim> (F);
              const Tensor<2,dim> sigma_ALE = NSE_in_ALE::get_stress_fluid_except_pressure_ALE<dim> (density_fluid, viscosity, grad_v, grad_v_T, F_Inverse, F_Inverse_T );
           
              Tensor<2,dim> stress_fluid;
              stress_fluid.clear();
              stress_fluid = (J * sigma_ALE * F_Inverse_T);
           
              Tensor<2,dim> fluid_pressure;
              fluid_pressure.clear();
              fluid_pressure = (-pI * J * F_Inverse_T);
           
              drag_lift_value -= (stress_fluid + fluid_pressure) * 
              fe_face_values.normal_vector(q_point)* fe_face_values.JxW(q_point);                     
            }
          }     
    }               
  } 

  drag_summed += 0.5 * std::abs(drag_lift_value[0]);
  // on the half channel
  drag = 0.5 * drag_lift_value[0];   

  if (compute_short_scale)
  {
    std::cout << "SScDrag:         " << timestep_number << "   " << time << "  " << drag_lift_value[0] << std::endl;
    std::cout << "SScLift:         " << timestep_number << "   " << time << "  " << drag_lift_value[1] << std::endl;
    std::cout << "SScDrag summed:  " << timestep_number << "   " << time << "  " << drag_summed << std::endl;
    std::cout << "SScDrag (half):  " << timestep_number << "   " << time << "  " << drag << std::endl;
  }
  else 
  {
    std::cout << "LScDrag:         " << timestep_number << "   " << time << "  " << drag_lift_value[0] << std::endl;
    std::cout << "LScLift:         " << timestep_number << "   " << time << "  " << drag_lift_value[1] << std::endl;
    std::cout << "LScDrag summed:  " << timestep_number << "   " << time << "  " << drag_summed << std::endl;
    std::cout << "LScDrag (half):  " << timestep_number << "   " << time << "  " << drag << std::endl;
  }
}





// Now, we arrive at the function that is responsible 
// to compute the line integrals for the drag and the lift. Note, that 
// by a proper transformation via the Gauss theorem, the both 
// quantities could also be achieved by domain integral computation. 
// Nevertheless, we choose the line integration because deal.II provides
// all routines for face value evaluation. 
template <int dim>
void FSI_ALE_Problem<dim>::compute_outflow()
{
    
  const QGauss<dim-1> face_quadrature_formula (3);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
            update_values | update_gradients | update_normal_vectors | 
            update_JxW_values);
  
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<long unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<Vector<double> >  face_solution_values (n_face_q_points, 
                  Vector<double> (number_coefficients));

  std::vector<std::vector<Tensor<1,dim> > > 
    face_solution_grads (n_face_q_points, std::vector<Tensor<1,dim> > (number_coefficients));
  
  double drag_lift_value = 0.0;
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  for (; cell!=endc; ++cell)
  {
    // First, we are going to compute the forces that
    // act on the cylinder. We notice that only the fluid 
    // equations are defined here.
    for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id()==1)
      {
        fe_face_values.reinit (cell, face);
        fe_face_values.get_function_values (solution, face_solution_values);
        fe_face_values.get_function_gradients (solution, face_solution_grads);
          
        for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
        {         
          const Tensor<1,dim> v = ALE_Transformations::get_v<dim> (q_point, face_solution_values);

          drag_lift_value += v * 
          fe_face_values.normal_vector(q_point)* fe_face_values.JxW(q_point);
        }
      } // end boundary 1 for fluid
  } 
   
  if (compute_short_scale)
  {
    std::cout << "SScOutflow:      " << timestep_number << "   " << time << "  " << drag_lift_value << std::endl;
  }
  else 
    std::cout << "LScOutflow:      " << timestep_number << "   " << time << "  " << drag_lift_value << std::endl;
}

// Here, we compute the four quantities of interest:
// the x and y-displacements of the structure, the drag, and the lift.
template<int dim>
void FSI_ALE_Problem<dim>::compute_functional_values()
{
  double x1,y1;
  x1 = compute_point_value(Point<dim>(0.0+growth_displacement,-1.0), dim);
  y1 = compute_point_value(Point<dim>(0.0+growth_displacement,-1.0), dim+1);

  
  std::cout << "-----------------------------------" << std::endl;
  std::cout << "DisX: " << timestep_number << "   " << time << "  " << x1 << std::endl; // x_displacement for Point
  std::cout << "DisY: " << timestep_number << "   " << time << "  " << y1 << std::endl; // y_displacement for Point
  std::cout << "UY:   " << timestep_number << "   " << time << "  " << u_y << std::endl; // abs of y_displacement for Point
  std::cout << "-----------------------------------" << std::endl;

  double x2,y2;
  x2 = compute_point_value(Point<dim>(0.0,-1.0), dim);
  y2 = compute_point_value(Point<dim>(0.0,-1.0), dim+1);

  
  std::cout << "-----------------------------------" << std::endl;
  std::cout << "DisX Middle: " << timestep_number << "   " << time << "  " << x2 << std::endl; // x_displacement for Point
  std::cout << "DisY Middle: " << timestep_number << "   " << time << "  " << y2 << std::endl; // y_displacement for Point
  std::cout << "UY Middle:   " << timestep_number << "   " << time << "  " << std::abs(y2) << std::endl; // abs of y_displacement for Point
  std::cout << "-----------------------------------" << std::endl;
  
  compute_outflow();
  compute_drag_lift_fsi_fluid_tensor();
  compute_vorticity();

  std::cout << "-----------------------------------" << std::endl;
  compute_line_velocity( 0. );
  std::cout << "-----------------------------------" << std::endl;
}


template<int dim>
void FSI_ALE_Problem<dim>::update_uy()
{
  u_y = std::abs(compute_point_value(Point<dim>(0.0+growth_displacement,-1.0), dim+1));
}

template<int dim>
void FSI_ALE_Problem<dim>::compute_vorticity()
{
  QGauss<dim>   quadrature_formula(degree+2);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);
  const unsigned int   n_q_points      = quadrature_formula.size();

  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, 
                std::vector<Tensor<1,dim> > (number_coefficients));
  
  double vorticity = 0.0;

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  
  for (; cell!=endc; ++cell)
  { 
    fe_values.reinit (cell);
      
    fe_values.get_function_gradients (solution, old_solution_grads);
      
    if (cell->material_id() == 0)
    {
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        const Tensor<2,dim> F = ALE_Transformations::get_F<dim> (q, old_solution_grads);
        
        const double J = ALE_Transformations::get_J<dim> (F);
        const double J_inverse = 1.0/J;

        const Tensor<2,dim> grad_u = ALE_Transformations ::get_grad_u<dim> (q, old_solution_grads);

        const Tensor<2,dim> grad_v = ALE_Transformations ::get_grad_v<dim> (q, old_solution_grads);
        
        Tensor<2,dim> JF;
        JF[0][0] = 1.0 + grad_u[1][1];
        JF[0][1] = - grad_u[0][1];
        JF[1][0] = - grad_u[1][0];
        JF[1][1] = 1.0 + grad_u[0][0];

        double v1y = grad_v[0][0] * JF[0][1] + grad_v[0][1] * JF[1][1]; 
        double v2x = grad_v[1][0] * JF[0][0] + grad_v[1][1] * JF[1][0]; 

        vorticity += J_inverse * (v1y - v2x) * (v1y - v2x)
                    * fe_values.JxW(q);  
      }
    }
  }
  
  if (compute_short_scale)
    std::cout << "SScVorticity:    " << timestep_number << "   " << time << "  " << vorticity << std::endl;
  else 
    std::cout << "LScVorticity:    " << timestep_number << "   " << time << "  " << vorticity << std::endl;
}


// As usual, we have to call the run method. It handles
// the output stream to the terminal.
// Second, we define some output skip that is necessary 
// (and really useful) to avoid to much printing 
// of solutions. For large time dependent problems it is 
// sufficient to print only each tenth solution. 
// Third, we perform the time stepping scheme of 
// the solution process.
template <int dim>
void FSI_ALE_Problem<dim>::run () 
{  
  setup_system();
  set_initial_condition( );

  std::cout << "\n==============================" 
      << "==========================================="  << std::endl;
  std::cout << "Parameters\n" 
      << "==========\n"
      << "Density fluid:     "   <<  density_fluid << "\n"
      << "Density structure: "   <<  density_structure << "\n"  
      << "Viscosity fluid:   "   <<  viscosity << "\n"
      << "alpha_u:           "   <<  alpha_u << "\n"
      << "alpha_us:          "   <<  alpha_us << "\n"  
      << "Lame coeff. mu:    "   <<  lame_coefficient_mu << "\n"
      << std::endl;

 
  const unsigned int output_skip = 1;
  drag_summed = 0.0;
  drag = growth_initial;
  u_y = 0.0;
  do
  { 
    compute_short_scale = 1.0;

    max_no_timesteps = 15000; //250;
    if (timestep_number < 1)
      timestep = 43200; //43200; //3600;//10800;//5400; //43200;//10800; //21600; //600;
    else 
      timestep = 43200; //43200; //3600;//10800; //5400; //43200;//10800; //21600; //600;//86400.0;

    std::cout << "Timestep " << timestep_number 
      << " (" << time_stepping_scheme << ") " 
      << "Short term scale: " <<  compute_short_scale
      <<    ": " << time
      << " (" << timestep << ")"
      << "\n==============================" 
      << "===========================================" 
      << std::endl; 
        
    std::cout << std::endl; 
        
    // Compute next time step
    old_timestep_solution = solution;
    newton_iteration (time);  

    // Compute functional values: dx, dy, drag, lift
    std::cout << std::endl;

    if (!compute_short_scale)
    {
      update_uy();
    }
        
    compute_functional_values();
    compute_minimal_J();
       
    // Write solutions 
    if ((timestep_number % output_skip == 0))
      output_results (timestep_number,solution);

    time += timestep;
    ++timestep_number;

    if (compute_short_scale)
    {
      drag_summed = 0.0;
      final_drag_summed = 0.0;
    }
    refine_grid();
  }
  while (timestep_number <= max_no_timesteps);
}

// The main function looks almost the same
// as in all other deal.II tuturial steps. 
int main () 
{
  std::signal(SIGSEGV, [](int){ std::cerr << "Segmentation fault triggered\n"; std::abort(); });
  try
  {
    deallog.depth_console (0);

    FSI_ALE_Problem<2> flow_problem(1);
    flow_problem.run ();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...) 
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
