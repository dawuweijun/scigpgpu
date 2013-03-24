/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/
#ifndef CONTEXT_HPP_
#define CONTEXT_HPP_

#include <vector>
#include <map>
#include <memory> // std::shared_ptr

/*!
 * This class handles all initialisation/uninitialisation step needed when using GPGPU.
 */
template<class ModeDefinition>
  class Context
  {
    enum
    {
      mode = ModeDefinition::mode
    };
    typedef typename ModeDefinition::Context_Handle Context_Handle;
    typedef typename ModeDefinition::Platform Platform;
    typedef typename ModeDefinition::Device_Handle Device_Handle;
    typedef typename ModeDefinition::Stream Stream;
    typedef typename ModeDefinition::Status Status;
    friend class Kernel<ModeDefinition> ;
  protected:
    Context_Handle cont;
    Platform* platforms;

    Device<ModeDefinition> current_device;
    std::vector<Device<ModeDefinition> > devices_list;
    std::map<std::string, Module<ModeDefinition> > loadedModule;

    static int
    number_of_device();
  public:
    inline const Device_Handle&
    get_dev() const
    {
      return current_device.dev;
    }

    Context();

    int initContext();

    /*!
     * Initialise device for computing.
     * This function must be called before any GPGPU computing take place.
     * set_current_device<false> configures device for classic computing
     * set_current_device<true> configures device to use a OpenGL-interoperable environment.
     */
    template<bool isGL>
    void set_current_device(const Device<ModeDefinition>& device);

    /*!
     * Returns a list of usable device on the platform.
     */
    inline const std::vector<Device<ModeDefinition> >&  get_devices_list()
    {
      return devices_list;
    }

    /*!
     * Eventually loads/builds and returns a reference to a module.
     */
    const Module<ModeDefinition>*  getModule(std::string filename)
    {
      if (loadedModule.find(filename) == loadedModule.end())
        {
          loadedModule[filename] = Module<ModeDefinition> (filename, cont, current_device.dev);
          loadedModule[filename].load();
          if(loadedModule[filename].isloaded == false)
          {
              return NULL;
          }
        }
      return &loadedModule[filename];
    }

    /*!
     *  Generate a queue for concurrent query on a device.
     */
    Queue<ModeDefinition>
    genQueue()
    {
      return Queue<ModeDefinition> (cont, current_device.dev);
    }

    ~Context();

    /*!
     *  Create a contigous zone of memory in device memory
     */
    template<typename T>
      inline std::shared_ptr<Matrix<ModeDefinition, T> >  genMatrix(Queue<ModeDefinition> q, int n, T* tmp = NULL)
      {
        return std::make_shared<Matrix<ModeDefinition, T> > (cont, q.stream, n, tmp);
      }

    template<typename T>
      inline std::shared_ptr<GLMatrix<ModeDefinition, T> >  genGLMatrix(Queue<ModeDefinition> q, GLuint vbo)
      {
        return std::make_shared<GLMatrix<ModeDefinition, T> > (cont, q.stream, vbo);
      }

    /*!
     *  Copy into a new contigous zone of memory in device memory
     */
    template<typename T>
      inline std::shared_ptr<Matrix<ModeDefinition, T> >  copyMatrix(const std::shared_ptr<Matrix<ModeDefinition, T> >& input)
      {
        return std::make_shared<Matrix<ModeDefinition, T> > (*input);
      }

  };

#endif /* CONTEXT_HPP_ */
